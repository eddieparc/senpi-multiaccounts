import { request } from "node:http";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { kiroProviderPackage } from "../src/providers/kiro/index.js";
import { KiroAuthError, type KiroTokens, type LoginCallbacks, loginKiro, refreshKiro } from "../src/providers/kiro/oauth.js";

type FetchArgs = { url: string; body: Record<string, unknown>; headers: Record<string, string> };

const originalFetch = globalThis.fetch;
const calls: FetchArgs[] = [];

function mockFetch(routes: Record<string, { status?: number; body: unknown }>): void {
	globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
		const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
		calls.push({
			url,
			body: init?.body ? JSON.parse(String(init.body)) : {},
			headers: (init?.headers ?? {}) as Record<string, string>,
		});
		const route = Object.entries(routes).find(([pattern]) => url.includes(pattern))?.[1];
		if (!route) throw new Error(`unexpected fetch: ${url}`);
		return new Response(JSON.stringify(route.body), {
			status: route.status ?? 200,
			headers: { "Content-Type": "application/json" },
		});
	}) as typeof fetch;
}

function callbacks(overrides: Partial<LoginCallbacks> = {}): LoginCallbacks {
	return {
		onAuth: vi.fn(),
		onPrompt: vi.fn(async () => ""),
		...overrides,
	};
}

function socialTokens(overrides: Partial<KiroTokens> = {}): KiroTokens {
	return {
		access: "old-access",
		refresh: "refresh-token",
		expires: 0,
		authMethod: "google",
		region: "us-east-1",
		...overrides,
	};
}

afterEach(() => {
	globalThis.fetch = originalFetch;
	calls.length = 0;
	vi.restoreAllMocks();
});

describe("social login", () => {
	it("exchanges a pasted callback for tokens and keeps the profile ARN", async () => {
		mockFetch({
			"/oauth/token": {
				body: {
					accessToken: "new-access",
					refreshToken: "new-refresh",
					expiresIn: 3600,
					profileArn: "arn:aws:codewhisperer:us-east-1:1:profile/ABC",
				},
			},
		});

		let authorizeUrl = "";
		const tokens = await loginKiro(
			callbacks({
				onAuth: (info) => {
					authorizeUrl = info.url;
				},
				onPrompt: async () => {
					const state = new URL(authorizeUrl).searchParams.get("state");
					return `http://localhost:3128/oauth/callback?code=auth-code&state=${state}`;
				},
			}),
			"google",
		);

		expect(tokens.access).toBe("new-access");
		expect(tokens.refresh).toBe("new-refresh");
		expect(tokens.authMethod).toBe("google");
		expect(tokens.profileArn).toBe("arn:aws:codewhisperer:us-east-1:1:profile/ABC");
		expect(tokens.expires).toBeGreaterThan(Date.now());

		const exchange = calls.find((call) => call.url.includes("/oauth/token"));
		expect(exchange?.body.code).toBe("auth-code");
		expect(exchange?.body.code_verifier).toEqual(expect.any(String));
	});

	it("sends a PKCE S256 challenge, never the verifier", async () => {
		mockFetch({ "/oauth/token": { body: { accessToken: "a", refreshToken: "r", expiresIn: 60 } } });

		let authorizeUrl = "";
		await loginKiro(
			callbacks({
				onAuth: (info) => {
					authorizeUrl = info.url;
				},
				onPrompt: async () => {
					const state = new URL(authorizeUrl).searchParams.get("state");
					return `http://localhost:3128/oauth/callback?code=c&state=${state}`;
				},
			}),
			"google",
		);

		const params = new URL(authorizeUrl).searchParams;
		expect(params.get("code_challenge_method")).toBe("S256");
		const challenge = params.get("code_challenge");
		const verifier = calls.find((call) => call.url.includes("/oauth/token"))?.body.code_verifier;
		expect(challenge).toBeTruthy();
		expect(challenge).not.toBe(verifier);
	});

	it("rejects a callback whose state does not match", async () => {
		mockFetch({ "/oauth/token": { body: {} } });

		await expect(
			loginKiro(
				callbacks({ onPrompt: async () => "http://localhost:3128/oauth/callback?code=c&state=attacker" }),
				"google",
			),
		).rejects.toThrow(/state did not match/);
		expect(calls.some((call) => call.url.includes("/oauth/token"))).toBe(false);
	});

	it("reports a denied authorization", async () => {
		mockFetch({ "/oauth/token": { body: {} } });
		await expect(
			loginKiro(callbacks({ onPrompt: async () => "http://localhost:3128/oauth/callback?error=access_denied" }), "github"),
		).rejects.toThrow(/denied/);
	});
});

describe("builder id login", () => {
	it("polls until authorization completes, then returns tokens with client metadata", async () => {
		mockFetch({
			"client/register": { body: { clientId: "cid", clientSecret: "secret" } },
			device_authorization: {
				// 1s is the smallest interval the flow honours; keeps the poll real but fast.
				body: { deviceCode: "dev", userCode: "USER-CODE", verificationUriComplete: "https://device.example", interval: 1, expiresIn: 60 },
			},
			"/token": { body: { accessToken: "idc-access", refreshToken: "idc-refresh", expiresIn: 3600 } },
		});

		const onDeviceCode = vi.fn();
		const tokens = await loginKiro(callbacks({ onDeviceCode }), "builder-id");

		expect(tokens.authMethod).toBe("builder-id");
		expect(tokens.access).toBe("idc-access");
		// Client metadata must persist or the credential can never refresh.
		expect(tokens.clientId).toBe("cid");
		expect(tokens.clientSecret).toBe("secret");
		expect(onDeviceCode).toHaveBeenCalledWith(expect.objectContaining({ userCode: "USER-CODE" }));
	});

	it("stops polling as soon as the login is aborted", async () => {
		mockFetch({
			"client/register": { body: { clientId: "cid", clientSecret: "secret" } },
			device_authorization: {
				body: { deviceCode: "dev", userCode: "CODE", verificationUriComplete: "https://device.example", interval: 30, expiresIn: 600 },
			},
			"/token": { body: { error: "authorization_pending" } },
		});

		const controller = new AbortController();
		const pending = loginKiro(callbacks({ signal: controller.signal }), "builder-id");
		// Abort while the flow is sleeping between polls; it must reject promptly
		// rather than waiting out the 30s interval.
		controller.abort();

		await expect(pending).rejects.toMatchObject({ name: "KiroAuthError", permanent: true });
	});
});

describe("method selection", () => {
	it("offers Google, GitHub and Builder ID", async () => {
		mockFetch({ "/oauth/token": { body: { accessToken: "a", refreshToken: "r", expiresIn: 60 } } });
		const onSelect = vi.fn(async (_prompt: { message: string; options: { id: string; label: string }[] }) => "google");

		let authorizeUrl = "";
		await loginKiro(
			callbacks({
				onSelect,
				onAuth: (info) => {
					authorizeUrl = info.url;
				},
				onPrompt: async () => {
					const state = new URL(authorizeUrl).searchParams.get("state");
					return `http://localhost:3128/oauth/callback?code=c&state=${state}`;
				},
			}),
		);

		const offered = onSelect.mock.calls[0]?.[0] as unknown as { options: { id: string }[] } | undefined;
		expect(offered?.options.map((option) => option.id)).toEqual(["google", "github", "builder-id"]);
	});

	it("treats a cancelled selection as permanent", async () => {
		await expect(loginKiro(callbacks({ onSelect: async () => undefined }))).rejects.toMatchObject({
			name: "KiroAuthError",
			permanent: true,
		});
	});
});

describe("refresh", () => {
	it("refreshes social tokens through the Kiro auth service", async () => {
		mockFetch({
			"/refreshToken": { body: { accessToken: "rotated", refreshToken: "rotated-refresh", expiresIn: 3600, profileArn: "arn:new" } },
		});

		const refreshed = await refreshKiro(socialTokens());

		expect(refreshed.access).toBe("rotated");
		expect(refreshed.refresh).toBe("rotated-refresh");
		expect(refreshed.profileArn).toBe("arn:new");
		expect(calls[0]?.url).toContain("/refreshToken");
	});

	it("keeps the existing refresh token when the service does not rotate it", async () => {
		mockFetch({ "/refreshToken": { body: { accessToken: "rotated", expiresIn: 3600 } } });
		const refreshed = await refreshKiro(socialTokens());
		expect(refreshed.refresh).toBe("refresh-token");
	});

	it("refreshes Builder ID tokens through SSO OIDC", async () => {
		mockFetch({ "oidc.us-east-1.amazonaws.com/token": { body: { accessToken: "idc-new", expiresIn: 3600 } } });

		const refreshed = await refreshKiro(
			socialTokens({ authMethod: "builder-id", clientId: "cid", clientSecret: "secret" }),
		);

		expect(refreshed.access).toBe("idc-new");
		expect(calls[0]?.url).toContain("oidc.us-east-1.amazonaws.com/token");
		expect(calls[0]?.body.grantType).toBe("refresh_token");
	});

	it("fails permanently when Builder ID client metadata is missing", async () => {
		await expect(refreshKiro(socialTokens({ authMethod: "builder-id" }))).rejects.toMatchObject({
			permanent: true,
		});
	});

	it("marks a rejected refresh token as permanent so the account blocks", async () => {
		mockFetch({ "/refreshToken": { status: 400, body: { error: "invalid_grant" } } });

		const error = await refreshKiro(socialTokens()).catch((caught: unknown) => caught);
		expect(error).toBeInstanceOf(KiroAuthError);
		expect((error as KiroAuthError).permanent).toBe(true);
	});

	it("treats a 429 during refresh as retryable", async () => {
		mockFetch({ "/refreshToken": { status: 429, body: { message: "slow down" } } });

		const error = await refreshKiro(socialTokens()).catch((caught: unknown) => caught);
		expect((error as KiroAuthError).permanent).toBe(false);
	});
});

/**
 * Installed host contract (LoginDialogComponent): one text-input resolver.
 * showPrompt replaces it without settling the previous promise, and submit
 * resolves only the current resolver. Messages already on screen stay there.
 */
function createLoginDialog() {
	const selectQueue: string[] = [];
	const selectWaiters: Array<(message: string) => void> = [];
	const promptQueue: string[] = [];
	const promptWaiters: Array<(message: string) => void> = [];
	let textResolver: ((value: string) => void) | null = null;
	let selectResolver: ((value: string | undefined) => void) | null = null;

	const deliver = (queue: string[], waiters: Array<(message: string) => void>, message: string) => {
		const waiter = waiters.shift();
		if (waiter) waiter(message);
		else queue.push(message);
	};
	const wait = (queue: string[], waiters: Array<(message: string) => void>) => {
		const queued = queue.shift();
		if (queued !== undefined) return Promise.resolve(queued);
		return new Promise<string>((resolve) => {
			waiters.push(resolve);
		});
	};

	return {
		onSelect(prompt: { message: string }) {
			deliver(selectQueue, selectWaiters, prompt.message);
			return new Promise<string | undefined>((resolve) => {
				selectResolver = resolve;
			});
		},
		onPrompt(prompt: { message: string }) {
			deliver(promptQueue, promptWaiters, prompt.message);
			return new Promise<string>((resolve) => {
				textResolver = resolve;
			});
		},
		waitForSelect: () => wait(selectQueue, selectWaiters),
		waitForPrompt: () => wait(promptQueue, promptWaiters),
		choose(id: string) {
			const resolve = selectResolver;
			if (!resolve) throw new Error("no select in progress");
			selectResolver = null;
			resolve(id);
		},
		submit(value: string) {
			const resolve = textResolver;
			if (!resolve) throw new Error("no text prompt in progress");
			textResolver = null;
			resolve(value);
		},
	};
}

function requestCallback(url: string): Promise<void> {
	return new Promise((resolve, reject) => {
		const req = request(url, (response) => {
			response.resume();
			response.on("end", () => resolve());
		});
		req.on("error", reject);
		req.end();
	});
}

function callbackUrlFrom(authorizeUrl: string, code = "auth-code"): string {
	const authorize = new URL(authorizeUrl);
	const state = authorize.searchParams.get("state");
	const redirect = authorize.searchParams.get("redirect_uri");
	if (!state || !redirect) throw new Error(`authorize URL missing state or redirect: ${authorizeUrl}`);
	const port = new URL(redirect).port;
	if (!port) throw new Error(`callback server did not publish a port: ${redirect}`);
	const callback = new URL("http://127.0.0.1");
	callback.port = port;
	callback.pathname = "/oauth/callback";
	callback.searchParams.set("code", code);
	callback.searchParams.set("state", state);
	return callback.toString();
}

function accountFrom(value: unknown): { name: string; access: string } {
	if (typeof value !== "object" || value === null || !("accounts" in value)) {
		throw new Error("login did not return an account pool");
	}
	const accounts = value.accounts;
	if (!Array.isArray(accounts) || accounts.length === 0) throw new Error("login returned no accounts");
	const account = accounts[0];
	if (typeof account !== "object" || account === null) throw new Error("login account is missing");
	const name = "name" in account ? account.name : undefined;
	const access = "access" in account ? account.access : undefined;
	if (typeof name !== "string" || typeof access !== "string") throw new Error("login account fields are not strings");
	return { name, access };
}

describe("callback prompt versus account name", () => {
	it("keeps a pasted callback on the authorization path", async () => {
		mockFetch({
			"/oauth/token": { body: { accessToken: "pasted-access", refreshToken: "pasted-refresh", expiresIn: 3600 } },
		});
		const agentDir = await mkdtemp(join(tmpdir(), "kiro-paste-"));
		const dialog = createLoginDialog();
		let authorizeUrl = "";
		const built = await kiroProviderPackage().build({ env: {}, agentDir });
		if (!built.oauth) throw new Error("kiro oauth login is not registered");

		const settled = built.oauth
			.login({
				onAuth: (info: { url: string }) => {
					authorizeUrl = info.url;
				},
				onPrompt: dialog.onPrompt,
				onSelect: dialog.onSelect,
				onDeviceCode: vi.fn(),
			})
			.then(
				(value) => ({ kind: "done" as const, value }),
				(error: unknown) => ({ kind: "error" as const, error }),
			);

		try {
			const selectWait = dialog.waitForSelect();
			await selectWait;
			let promptWait = dialog.waitForPrompt();
			dialog.choose("google");

			for (;;) {
				const outcome = await Promise.race([
					promptWait.then((message) => ({ kind: "prompt" as const, message })),
					settled,
				]);
				if (outcome.kind !== "prompt") {
					expect(outcome.kind).toBe("done");
					if (outcome.kind === "done") expect(accountFrom(outcome.value).name).toBe("work");
					break;
				}
				promptWait = dialog.waitForPrompt();
				if (!authorizeUrl) dialog.submit("work");
				else dialog.submit(callbackUrlFrom(authorizeUrl, "pasted-code"));
			}

			const exchange = calls.find((call) => call.url.includes("/oauth/token"));
			expect(exchange?.body.code).toBe("pasted-code");
		} finally {
			await settled;
			await rm(agentDir, { recursive: true, force: true });
		}
	});

	it("does not accept the callback URL as the account name when the local server wins", async () => {
		mockFetch({
			"/oauth/token": { body: { accessToken: "server-access", refreshToken: "server-refresh", expiresIn: 3600 } },
		});
		const agentDir = await mkdtemp(join(tmpdir(), "kiro-race-"));
		const dialog = createLoginDialog();
		let authorizeUrl = "";
		const built = await kiroProviderPackage().build({ env: {}, agentDir });
		if (!built.oauth) throw new Error("kiro oauth login is not registered");

		const settled = built.oauth
			.login({
				onAuth: (info: { url: string }) => {
					authorizeUrl = info.url;
				},
				onPrompt: dialog.onPrompt,
				onSelect: dialog.onSelect,
				onDeviceCode: vi.fn(),
			})
			.then(
				(value) => ({ kind: "done" as const, value }),
				(error: unknown) => ({ kind: "error" as const, error }),
			);

		try {
			await dialog.waitForSelect();
			let promptWait = dialog.waitForPrompt();
			dialog.choose("google");

			await promptWait;
			if (!authorizeUrl) {
				promptWait = dialog.waitForPrompt();
				dialog.submit("work");
				await promptWait;
			}
			expect(authorizeUrl).not.toBe("");

			const afterCapture = dialog.waitForPrompt();
			await requestCallback(callbackUrlFrom(authorizeUrl, "server-code"));
			const outcome = await Promise.race([
				afterCapture.then((promptMessage) => ({ kind: "prompt" as const, promptMessage })),
				settled,
			]);

			if (outcome.kind === "prompt") {
				// The paste instruction is still on screen. The user submits the URL
				// to whatever prompt now owns the single resolver.
				dialog.submit(callbackUrlFrom(authorizeUrl, "server-code"));
				const result = await settled;
				const errorMessage = result.kind === "error" && result.error instanceof Error ? result.error.message : "";
				expect(errorMessage).not.toMatch(/Invalid account name/);
			}

			expect(outcome.kind).toBe("done");
			if (outcome.kind === "done") {
				expect(accountFrom(outcome.value).name).toBe("work");
				expect(accountFrom(outcome.value).access).toBe("server-access");
			}
			const exchange = calls.find((call) => call.url.includes("/oauth/token"));
			expect(exchange?.body.code).toBe("server-code");
		} finally {
			await settled;
			await rm(agentDir, { recursive: true, force: true });
		}
	});
});
