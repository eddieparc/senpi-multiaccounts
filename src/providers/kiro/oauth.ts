import { createHash, randomBytes } from "node:crypto";
import { createServer, type Server } from "node:http";
import {
	KIRO_AUTH_METHOD_LABELS,
	type KiroAuthMethod,
	KIRO_IDC,
	KIRO_REGION,
	KIRO_SOCIAL,
} from "./config.js";

/**
 * Kiro OAuth.
 *
 * Two distinct backends, because Kiro's service only federates social logins:
 *   - Google / GitHub  -> Kiro desktop auth service, PKCE + localhost callback.
 *   - AWS Builder ID   -> AWS SSO OIDC device-code flow.
 * Each mints its own credential shape and refreshes through its own endpoint.
 */

export interface KiroTokens {
	access: string;
	refresh: string;
	/** Epoch millis. */
	expires: number;
	authMethod: KiroAuthMethod;
	/** CodeWhisperer profile ARN issued with social tokens. */
	profileArn?: string;
	region: string;
	/** IdC only: needed to refresh through SSO OIDC. */
	clientId?: string;
	clientSecret?: string;
	/** Account email, when the provider discloses it. */
	email?: string;
}

export interface LoginCallbacks {
	onAuth(info: { url: string; instructions?: string }): void;
	onDeviceCode?(info: { userCode: string; verificationUri: string }): void;
	onProgress?(message: string): void;
	onPrompt(prompt: { message: string; placeholder?: string }): Promise<string>;
	onSelect?(prompt: {
		message: string;
		options: { id: string; label: string }[];
	}): Promise<string | undefined>;
	signal?: AbortSignal;
}

const DEFAULT_EXPIRES_IN_SECONDS = 3_600;
const REQUEST_TIMEOUT_MS = 300_000;
const CALLBACK_PORT_SPAN = 20;
const SOCIAL_IDP: Record<"google" | "github", string> = { google: "Google", github: "Github" };

export class KiroAuthError extends Error {
	readonly permanent: boolean;
	readonly status?: number;

	constructor(message: string, options: { permanent?: boolean; status?: number; cause?: unknown } = {}) {
		super(message, options.cause === undefined ? undefined : { cause: options.cause });
		this.name = "KiroAuthError";
		this.permanent = options.permanent ?? false;
		if (options.status !== undefined) this.status = options.status;
	}
}

type JsonRecord = Record<string, unknown>;

function str(value: unknown): string | undefined {
	return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function pick(body: JsonRecord, ...keys: string[]): string | undefined {
	for (const key of keys) {
		const value = str(body[key]);
		if (value) return value;
	}
	return undefined;
}

function expiresAt(body: JsonRecord, now: number): number {
	const explicit = pick(body, "expiresAt", "expires_at");
	if (explicit) {
		const parsed = Date.parse(explicit);
		if (Number.isFinite(parsed) && parsed > now) return parsed;
	}
	const seconds = body.expiresIn ?? body.expires_in;
	const value = typeof seconds === "number" && Number.isFinite(seconds) && seconds > 0 ? seconds : DEFAULT_EXPIRES_IN_SECONDS;
	return now + value * 1_000;
}

async function postJson(
	url: string,
	payload: unknown,
	options: { headers?: Record<string, string>; signal?: AbortSignal; accept?: (body: JsonRecord, response: Response) => boolean } = {},
): Promise<JsonRecord> {
	const controller = new AbortController();
	const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
	timeout.unref?.();
	const onAbort = () => controller.abort();
	options.signal?.addEventListener("abort", onAbort, { once: true });

	let response: Response;
	try {
		response = await fetch(url, {
			method: "POST",
			headers: { "Content-Type": "application/json", Accept: "application/json", ...options.headers },
			body: JSON.stringify(payload),
			signal: controller.signal,
		});
	} catch (error) {
		throw new KiroAuthError(`Kiro auth request to ${url} failed`, { cause: error });
	} finally {
		clearTimeout(timeout);
		options.signal?.removeEventListener("abort", onAbort);
	}

	let body: JsonRecord = {};
	try {
		const parsed: unknown = await response.json();
		if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) body = parsed as JsonRecord;
	} catch {
		// Non-JSON body: keep the empty record and let status drive the error.
	}

	const accepted = options.accept ? options.accept(body, response) : response.ok;
	if (!accepted) {
		const detail = pick(body, "error_description", "message", "error") ?? `HTTP ${response.status}`;
		// 4xx other than 429 means the grant itself is bad; retrying cannot help.
		const permanent = response.status >= 400 && response.status < 500 && response.status !== 429;
		throw new KiroAuthError(`Kiro auth failed: ${detail}`, { status: response.status, permanent });
	}
	return body;
}

function base64Url(bytes: number): string {
	return randomBytes(bytes).toString("base64url");
}

function pkce(): { verifier: string; challenge: string; state: string } {
	const verifier = base64Url(32);
	return {
		verifier,
		challenge: createHash("sha256").update(verifier).digest("base64url"),
		state: base64Url(32),
	};
}

interface CallbackServer {
	redirectBaseUri: string;
	callbackUrl: string;
	wait(): Promise<string | null>;
	cancel(): void;
	close(): Promise<void>;
}

/**
 * Bind a localhost callback server so the browser redirect is captured without
 * the user copying a URL. Returns null when no port in the span is free, in
 * which case login falls back to manual paste.
 */
async function startCallbackServer(authMethod: "google" | "github"): Promise<CallbackServer | null> {
	const redirect = new URL(KIRO_SOCIAL.portalRedirectUri);
	const firstPort = Number.parseInt(redirect.port || "3128", 10);
	let settle: ((value: string | null) => void) | null = null;
	let settled = false;
	let boundPort = firstPort;

	const received = new Promise<string | null>((resolve) => {
		settle = (value) => {
			if (settled) return;
			settled = true;
			resolve(value);
		};
	});

	const server = createServer((request, response) => {
		const requestUrl = new URL(request.url ?? "/", `http://localhost:${boundPort}`);
		if (requestUrl.pathname !== KIRO_SOCIAL.callbackPath) {
			response.writeHead(404, { "Content-Type": "text/plain", Connection: "close" });
			response.end("Not found");
			return;
		}
		settle?.(`http://${redirect.hostname}:${boundPort}${request.url ?? ""}`);
		response.writeHead(200, { "Content-Type": "text/html; charset=utf-8", Connection: "close" });
		response.end(
			"<!doctype html><meta charset=utf-8><title>Kiro sign-in complete</title>" +
				"<h1>Kiro sign-in complete</h1><p>You can close this tab and return to senpi.</p>",
		);
	});

	const listen = (port: number): Promise<void> =>
		new Promise((resolve, reject) => {
			const onError = (error: Error) => {
				server.off("error", onError);
				reject(error);
			};
			server.once("error", onError);
			server.listen(port, "127.0.0.1", () => {
				server.off("error", onError);
				resolve();
			});
		});

	for (let port = firstPort; port <= firstPort + CALLBACK_PORT_SPAN; port++) {
		try {
			// Bind the IPv4 loopback explicitly. Binding by hostname can land on
			// `[::1]` while the browser resolves `localhost` to `127.0.0.1`, and the
			// redirect then never reaches this server.
			await listen(port);
			const address = server.address();
			boundPort = typeof address === "object" && address ? address.port : port;
			const redirectBaseUri = `${redirect.protocol}//${redirect.hostname}:${boundPort}`;
			const callbackUrl = new URL(KIRO_SOCIAL.callbackPath, `${redirectBaseUri}/`);
			callbackUrl.searchParams.set("login_option", authMethod);
			return {
				redirectBaseUri,
				callbackUrl: callbackUrl.toString(),
				wait: () => received,
				cancel: () => settle?.(null),
				close: () => new Promise<void>((resolve) => server.close(() => resolve())),
			};
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code !== "EADDRINUSE") break;
		}
	}

	await new Promise<void>((resolve) => server.close(() => resolve()));
	return null;
}

function parseCallback(input: string, expectedState: string): string {
	let url: URL;
	try {
		url = new URL(input.trim());
	} catch {
		throw new KiroAuthError("Kiro sign-in needs the full callback URL", { permanent: true });
	}

	const error = url.searchParams.get("error");
	if (error) throw new KiroAuthError(`Kiro sign-in was denied: ${error}`, { permanent: true });

	const states = url.searchParams.getAll("state");
	if (states.length !== 1 || states[0] !== expectedState) {
		// A mismatched state is the CSRF signal; never exchange such a code.
		throw new KiroAuthError("Kiro sign-in state did not match; start the login again", { permanent: true });
	}

	const codes = url.searchParams.getAll("code");
	const code = codes.length === 1 ? str(codes[0]) : undefined;
	if (!code) throw new KiroAuthError("Kiro sign-in callback carried no authorization code", { permanent: true });
	return code;
}

async function loginSocial(authMethod: "google" | "github", callbacks: LoginCallbacks): Promise<KiroTokens> {
	const { verifier, challenge, state } = pkce();
	const server = await startCallbackServer(authMethod);

	try {
		const redirectUri = server?.callbackUrl ?? KIRO_SOCIAL.redirectUri;
		const authorizeUrl = server
			? (() => {
					const url = new URL(KIRO_SOCIAL.portalUrl);
					url.searchParams.set("code_challenge", challenge);
					url.searchParams.set("code_challenge_method", "S256");
					url.searchParams.set("state", state);
					url.searchParams.set("redirect_uri", server.redirectBaseUri);
					url.searchParams.set("redirect_from", "kirocli");
					return url.toString();
				})()
			: (() => {
					const url = new URL(KIRO_SOCIAL.authorizeUrl);
					url.searchParams.set("idp", SOCIAL_IDP[authMethod]);
					url.searchParams.set("redirect_uri", redirectUri);
					url.searchParams.set("code_challenge", challenge);
					url.searchParams.set("code_challenge_method", "S256");
					url.searchParams.set("state", state);
					// Force the account chooser so a second account can be added
					// without first signing out of the first one.
					url.searchParams.set("prompt", "select_account");
					return url.toString();
				})();

		callbacks.onAuth({
			url: authorizeUrl,
			instructions: server
				? `Sign in to Kiro with ${KIRO_AUTH_METHOD_LABELS[authMethod]}. senpi captures the callback automatically; paste the callback URL only if that fails.`
				: `Sign in to Kiro with ${KIRO_AUTH_METHOD_LABELS[authMethod]}, then paste the full callback URL.`,
		});
		callbacks.onProgress?.("Waiting for the Kiro sign-in callback...");

		let callbackInput: string;
		if (server) {
			// Race the captured redirect against a manual paste so a blocked
			// browser redirect never strands the login. If the server wins, this
			// onPrompt stays pending and remains the host dialog's current text
			// input. Do not call onPrompt again after loginKiro returns: a second
			// prompt overwrites that single resolver and consumes the paste.
			const manual = callbacks
				.onPrompt({ message: "Paste the Kiro callback URL (only if the browser did not return automatically)", placeholder: server.callbackUrl })
				.then((value) => ({ source: "manual" as const, value }));
			const captured = server.wait().then((value) => (value ? { source: "server" as const, value } : null));
			const winner = await Promise.race([captured, manual]);
			if (winner?.source === "manual") server.cancel();
			callbackInput = winner?.value ?? (await server.wait()) ?? "";
		} else {
			callbackInput = await callbacks.onPrompt({ message: `Paste the full Kiro ${KIRO_AUTH_METHOD_LABELS[authMethod]} callback URL` });
		}

		const code = parseCallback(callbackInput, state);
		callbacks.onProgress?.("Exchanging the Kiro authorization code...");

		const now = Date.now();
		const body = await postJson(
			KIRO_SOCIAL.tokenUrl,
			{ code, code_verifier: verifier, redirect_uri: redirectUri },
			{ headers: { "User-Agent": "Kiro-CLI" } },
		);

		const access = pick(body, "accessToken", "access_token");
		const refresh = pick(body, "refreshToken", "refresh_token");
		if (!access || !refresh) throw new KiroAuthError("Kiro token exchange returned no tokens");

		const tokens: KiroTokens = {
			access,
			refresh,
			expires: expiresAt(body, now),
			authMethod,
			region: KIRO_REGION,
		};
		const profileArn = pick(body, "profileArn", "profile_arn");
		if (profileArn) tokens.profileArn = profileArn;
		return tokens;
	} finally {
		await server?.close().catch(() => undefined);
	}
}

function oidcEndpoint(region: string, path: string): string {
	return `https://oidc.${region}.amazonaws.com/${path}`;
}

/**
 * Sleep, rejecting immediately if the login is aborted. The abort listener is
 * always detached so repeated polling cannot accumulate listeners on a
 * long-lived signal.
 */
function wait(ms: number, signal?: AbortSignal): Promise<void> {
	return new Promise<void>((resolve, reject) => {
		const cancelled = () => new KiroAuthError("Kiro sign-in was cancelled", { permanent: true });
		if (signal?.aborted) {
			reject(cancelled());
			return;
		}
		const onAbort = () => {
			clearTimeout(timer);
			reject(cancelled());
		};
		const timer = setTimeout(() => {
			signal?.removeEventListener("abort", onAbort);
			resolve();
		}, ms);
		signal?.addEventListener("abort", onAbort, { once: true });
	});
}

async function loginBuilderId(callbacks: LoginCallbacks): Promise<KiroTokens> {
	const registration = await postJson(oidcEndpoint(KIRO_REGION, "client/register"), {
		clientName: KIRO_IDC.clientName,
		clientType: KIRO_IDC.clientType,
		scopes: KIRO_IDC.scopes,
		grantTypes: KIRO_IDC.grantTypes,
		issuerUrl: KIRO_IDC.issuerUrl,
	});
	const clientId = pick(registration, "clientId", "client_id");
	const clientSecret = pick(registration, "clientSecret", "client_secret");
	if (!clientId || !clientSecret) throw new KiroAuthError("Kiro client registration returned no client credentials");

	const device = await postJson(oidcEndpoint(KIRO_REGION, "device_authorization"), {
		clientId,
		clientSecret,
		startUrl: KIRO_IDC.startUrl,
	});
	const deviceCode = pick(device, "deviceCode", "device_code");
	const userCode = pick(device, "userCode", "user_code");
	const verificationUri =
		pick(device, "verificationUriComplete", "verification_uri_complete") ??
		pick(device, "verificationUri", "verification_uri");
	if (!deviceCode || !userCode || !verificationUri) {
		throw new KiroAuthError("Kiro device authorization was incomplete");
	}

	callbacks.onDeviceCode?.({ userCode, verificationUri });
	callbacks.onAuth({ url: verificationUri, instructions: `Approve the Kiro sign-in and enter code ${userCode}.` });

	const intervalSeconds = typeof device.interval === "number" && device.interval > 0 ? device.interval : 5;
	const expiresInSeconds = typeof device.expiresIn === "number" && device.expiresIn > 0 ? device.expiresIn : 600;
	let intervalMs = intervalSeconds * 1_000;
	const deadline = Date.now() + expiresInSeconds * 1_000;

	while (Date.now() < deadline) {
		await wait(intervalMs, callbacks.signal);

		const now = Date.now();
		const token = await postJson(
			oidcEndpoint(KIRO_REGION, "token"),
			{ clientId, clientSecret, deviceCode, grantType: "urn:ietf:params:oauth:grant-type:device_code" },
			{ accept: (body, response) => response.ok || body.error === "authorization_pending" || body.error === "slow_down" },
		);

		const access = pick(token, "accessToken", "access_token");
		if (access) {
			const refresh = pick(token, "refreshToken", "refresh_token");
			if (!refresh) throw new KiroAuthError("Kiro returned no refresh token");
			return {
				access,
				refresh,
				expires: expiresAt(token, now),
				authMethod: "builder-id",
				region: KIRO_REGION,
				clientId,
				clientSecret,
			};
		}

		if (token.error === "slow_down") intervalMs += intervalSeconds * 1_000;
		callbacks.onProgress?.("Waiting for the Kiro device authorization...");
	}

	throw new KiroAuthError("Kiro device code expired before it was approved", { permanent: true });
}

export interface KiroUsage {
	email?: string;
	usedCount: number;
	limitCount: number;
	/** Subscription tier, e.g. "KIRO PRO MAX". */
	plan?: string;
	/** Epoch millis when the allowance resets. */
	resetAt?: number;
}

/**
 * Read the account's credit allowance.
 *
 * The response carries per-resource rows in `usageBreakdownList`, *not*
 * top-level `usedCount`/`limitCount` fields. Reading only the top level yielded
 * 0/0 for every account and made this look like an unmetered plan, which
 * silently disabled usage-aware placement. The real numbers live in the `CREDIT`
 * row's `*WithPrecision` fields, which match what the Kiro account page shows
 * (e.g. 0.58 used / 5000 covered, resetting on the 1st).
 *
 * Drives both the usage dashboard and usage-aware placement. Callers treat a
 * throw as "unknown headroom" rather than an error, so this never blocks a
 * request.
 */
export async function fetchKiroUsage(
	tokens: Pick<KiroTokens, "access" | "region">,
	fetchImpl: typeof fetch = fetch,
): Promise<KiroUsage> {
	const region = tokens.region || KIRO_REGION;
	const url = new URL(`https://q.${region}.amazonaws.com/getUsageLimits`);
	url.searchParams.set("isEmailRequired", "true");
	url.searchParams.set("origin", "AI_EDITOR");
	url.searchParams.set("resourceType", "AGENTIC_REQUEST");

	const response = await fetchImpl(url, {
		headers: {
			authorization: `Bearer ${tokens.access}`,
			"x-amzn-kiro-agent-mode": "vibe",
			"amz-sdk-request": "attempt=1; max=1",
		},
	});
	if (!response.ok) throw new KiroAuthError(`Kiro usage request failed (HTTP ${response.status})`, { status: response.status });

	const body = (await response.json()) as JsonRecord;

	const num = (record: JsonRecord, ...names: string[]): number | undefined => {
		for (const name of names) {
			const value = record[name];
			if (typeof value === "number" && Number.isFinite(value)) return value;
		}
		return undefined;
	};

	// Prefer the CREDIT row; fall back to the first row so a renamed resource type
	// still yields numbers rather than silently reporting an unmetered plan.
	const rows = Array.isArray(body.usageBreakdownList) ? (body.usageBreakdownList as JsonRecord[]) : [];
	const row =
		rows.find((entry) => str(entry.resourceType) === "CREDIT") ??
		rows.find((entry) => num(entry, "usageLimitWithPrecision", "usageLimit") !== undefined) ??
		rows[0];

	const used = row ? num(row, "currentUsageWithPrecision", "currentUsage") : undefined;
	const limit = row ? num(row, "usageLimitWithPrecision", "usageLimit") : undefined;

	const userInfo = body.userInfo;
	const email =
		typeof userInfo === "object" && userInfo !== null ? str((userInfo as JsonRecord).email) : undefined;

	const subscription = body.subscriptionInfo;
	const plan =
		typeof subscription === "object" && subscription !== null
			? str((subscription as JsonRecord).subscriptionTitle)
			: undefined;

	// `nextDateReset` is epoch *seconds* (and arrives in exponential notation).
	const resetSeconds = row ? num(row, "nextDateReset") : undefined;
	const resetAt = resetSeconds ?? num(body, "nextDateReset");

	const usage: KiroUsage = {
		usedCount: used ?? 0,
		limitCount: limit ?? 0,
	};
	if (email) usage.email = email;
	if (plan) usage.plan = plan;
	if (resetAt !== undefined) usage.resetAt = Math.round(resetAt * 1000);
	return usage;
}

/** Run the Kiro login flow, asking which sign-in method to use. */
export async function loginKiro(callbacks: LoginCallbacks, method?: KiroAuthMethod): Promise<KiroTokens> {
	let authMethod = method;
	if (!authMethod) {
		const selected = await callbacks.onSelect?.({
			message: "Choose a Kiro sign-in method",
			options: (Object.keys(KIRO_AUTH_METHOD_LABELS) as KiroAuthMethod[]).map((id) => ({
				id,
				label: KIRO_AUTH_METHOD_LABELS[id],
			})),
		});
		if (selected === undefined) throw new KiroAuthError("Kiro sign-in was cancelled", { permanent: true });
		if (!(selected in KIRO_AUTH_METHOD_LABELS)) {
			throw new KiroAuthError(`Unsupported Kiro sign-in method '${selected}'`, { permanent: true });
		}
		authMethod = selected as KiroAuthMethod;
	}
	return authMethod === "builder-id" ? loginBuilderId(callbacks) : loginSocial(authMethod, callbacks);
}

/** Refresh Kiro tokens through whichever backend issued them. */
export async function refreshKiro(tokens: KiroTokens): Promise<KiroTokens> {
	if (!tokens.refresh) throw new KiroAuthError("Kiro credential has no refresh token", { permanent: true });
	const now = Date.now();

	if (tokens.authMethod === "builder-id") {
		if (!tokens.clientId || !tokens.clientSecret) {
			throw new KiroAuthError("Kiro Builder ID credential is missing its client metadata; sign in again", {
				permanent: true,
			});
		}
		const body = await postJson(oidcEndpoint(tokens.region || KIRO_REGION, "token"), {
			clientId: tokens.clientId,
			clientSecret: tokens.clientSecret,
			refreshToken: tokens.refresh,
			grantType: "refresh_token",
		});
		const access = pick(body, "accessToken", "access_token");
		if (!access) throw new KiroAuthError("Kiro refresh returned no access token");
		return {
			...tokens,
			access,
			refresh: pick(body, "refreshToken", "refresh_token") ?? tokens.refresh,
			expires: expiresAt(body, now),
		};
	}

	const body = await postJson(
		KIRO_SOCIAL.refreshUrl,
		{ refreshToken: tokens.refresh },
		{ headers: { "User-Agent": "Kiro-CLI" } },
	);
	const access = pick(body, "accessToken", "access_token");
	if (!access) throw new KiroAuthError("Kiro refresh returned no access token");

	const refreshed: KiroTokens = {
		...tokens,
		access,
		refresh: pick(body, "refreshToken", "refresh_token") ?? tokens.refresh,
		expires: expiresAt(body, now),
	};
	const profileArn = pick(body, "profileArn", "profile_arn");
	if (profileArn) refreshed.profileArn = profileArn;
	return refreshed;
}
