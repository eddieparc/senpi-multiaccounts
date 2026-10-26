import {
	addAccount,
	assertValidAccountName,
	isBlocked,
	pinAccount,
	removeAccount,
	unblockAccount,
	unpinAccount,
} from "../../core/accounts.js";
import type { AccountPoolState, MigrationPolicy } from "../../core/accounts.js";
import { DEFAULT_SCHEDULING_MODE, type SchedulingMode } from "../../core/affinity.js";
import { migrationSink } from "../../core/migration-sink.js";
import { emptyPool, readPool, type StoredPool, updatePool } from "../../core/store.js";
import type { ProviderBuildContext, ProviderPackage } from "../../core/types.js";
import { createAccountUsageCache, type AccountUsageCache } from "../../core/usage-detail-cache.js";
import { type AccountUsage, clampFraction, headroomOf, syncQuotaBlock } from "../../core/usage-window.js";
import { KIRO_AUTH_METHOD_LABELS, type KiroAuthMethod, KIRO_PROVIDER_ID } from "./config.js";
import { fetchKiroUsage, type KiroTokens, loginKiro, refreshKiro } from "./oauth.js";
import {
	buildKiroProviderConfig,
	type KiroProviderDeps,
	readKiroHeadroom,
	slotToTokens,
	tokensToSlot,
} from "./provider.js";

export { KIRO_PROVIDER_ID, KIRO_MODELS, resolveModels } from "./config.js";
export { loginKiro, refreshKiro, fetchKiroUsage, KiroAuthError, type KiroTokens } from "./oauth.js";
export { buildKiroProviderConfig, createKiroStreamSimple, tokensToSlot, slotToTokens } from "./provider.js";

/**
 * senpi's `/login <provider>` calls `oauth.login()` and persists whatever it
 * returns as that provider's single credential. Stock's Claude Agent SDK uses
 * this to manage a whole account pool, and we follow the same contract: the
 * credential *is* the pool, so `/login kiro` becomes the account manager and no
 * senpi-side change is needed to add or remove accounts.
 */

type LoginCallbacks = {
	onAuth(info: { url: string; instructions?: string }): void;
	onDeviceCode?(info: { userCode: string; verificationUri: string }): void;
	onProgress?(message: string): void;
	onPrompt(prompt: { message: string; placeholder?: string }): Promise<string>;
	onSelect?(prompt: {
		message: string;
		options: { id: string; label: string }[];
	}): Promise<string | undefined>;
	signal?: AbortSignal;
};

function toStored(state: AccountPoolState): StoredPool {
	const pool = emptyPool();
	pool.accounts = state.accounts;
	if (state.pinned !== undefined) pool.pinned = state.pinned;
	if (state.mode !== undefined) pool.mode = state.mode;
	if (state.bindings !== undefined) pool.bindings = state.bindings;
	if (state.cursor !== undefined) pool.cursor = state.cursor;
	if (state.migration !== undefined) pool.migration = state.migration;
	return pool;
}

function describe(state: AccountPoolState, now: number): string {
	if (state.accounts.length === 0) return "No Kiro accounts yet.";
	return state.accounts
		.map((slot) => {
			const marks: string[] = [];
			if (state.pinned === slot.name) marks.push("pinned");
			if (slot.blockReason === "auth_error") marks.push("needs re-login");
			else if (isBlocked(slot, now)) {
				marks.push(`blocked ${Math.ceil(((slot.blockedUntil ?? now) - now) / 1000)}s`);
			} else marks.push("available");
			const meta = slot.meta as { authMethod?: string; email?: string } | undefined;
			if (meta?.authMethod) marks.push(meta.authMethod);
			const label = meta?.email ? `${slot.name} <${meta.email}>` : slot.name;
			return `${label} — ${marks.join(", ")}`;
		})
		.join("\n");
}

function nextDefaultName(state: AccountPoolState): string {
	return state.accounts.length === 0 ? "default" : `account-${state.accounts.length + 1}`;
}

async function addFlow(state: AccountPoolState, callbacks: LoginCallbacks): Promise<AccountPoolState> {
	const method = await callbacks.onSelect?.({
		message: "Kiro sign-in method",
		options: (Object.keys(KIRO_AUTH_METHOD_LABELS) as KiroAuthMethod[]).map((id) => ({
			id,
			label: KIRO_AUTH_METHOD_LABELS[id],
		})),
	});
	if (method === undefined) throw new Error("Kiro sign-in cancelled");

	// Name before loginKiro. The paste prompt stays pending when the local
	// callback server wins its race, and the host dialog has one text resolver.
	// A later onPrompt would take the URL pasted into that still-open prompt.
	const suggested = nextDefaultName(state);
	const answer = (
		await callbacks.onPrompt({
			message:
				state.accounts.length === 0
					? "Name for this account"
					: `Name for this account (existing: ${state.accounts.map((s) => s.name).join(", ")})`,
			placeholder: suggested,
		})
	).trim();
	const name = answer || suggested;
	assertValidAccountName(name);

	const tokens = await loginKiro(callbacks as never, method as KiroAuthMethod);

	// Best-effort: the email makes the account list far easier to read, but a
	// failed lookup must not lose a successful login.
	let enriched: KiroTokens = tokens;
	try {
		const usage = await fetchKiroUsage(tokens);
		if (usage.email) enriched = { ...tokens, email: usage.email };
	} catch {
		// Usage endpoint unavailable; keep the token as-is.
	}

	return addAccount(state, tokensToSlot(name, enriched));
}

async function pickAccount(
	state: AccountPoolState,
	callbacks: LoginCallbacks,
	message: string,
): Promise<string | undefined> {
	if (state.accounts.length === 0) return undefined;
	return callbacks.onSelect?.({
		message,
		options: state.accounts.map((slot) => ({ id: slot.name, label: slot.name })),
	});
}

async function readKiroPoolUsage(
	context: ProviderBuildContext,
	cache: AccountUsageCache,
): Promise<Record<string, AccountUsage | undefined>> {
	const pool = readPool(context.agentDir, KIRO_PROVIDER_ID);
	const entries = await Promise.all(
		pool.accounts.map(async (slot): Promise<readonly [string, AccountUsage | undefined]> => {
			let tokens = slotToTokens(slot);
			if (Date.now() >= slot.expires) {
				try {
					tokens = await refreshKiro(tokens);
					updatePool(context.agentDir, KIRO_PROVIDER_ID, (state) => ({
						...state,
						accounts: state.accounts.map((candidate) =>
							candidate.name === slot.name ? tokensToSlot(slot.name, tokens, slot.source) : candidate,
						),
					}));
				} catch {
					return [slot.name, undefined];
				}
			}

			try {
				const result = await cache.get(slot.name, tokens.access, async () => {
					const usage = await fetchKiroUsage(tokens);
					const detail: AccountUsage = {
						windows:
							usage.limitCount > 0
								? [
										{
											label: "month",
											usedFraction: clampFraction((usage.usedCount / usage.limitCount) * 100),
											...(usage.resetAt === undefined ? {} : { resetsAt: usage.resetAt }),
										},
									]
								: [],
					};
					if (usage.plan) detail.plan = usage.plan;
					if (usage.email) detail.email = usage.email;
					return detail;
				});
				return [slot.name, result];
			} catch {
				return [slot.name, undefined];
			}
		}),
	);
	const detail = Object.fromEntries(entries);
	updatePool(context.agentDir, KIRO_PROVIDER_ID, (state) => ({
		...state,
		accounts: state.accounts.map((slot) => syncQuotaBlock(slot, detail[slot.name])),
	}));
	return detail;
}

/**
 * Menu-driven account manager. Every registration and removal path is reachable
 * from `/login kiro`, which is what makes native `/login` sufficient.
 */
async function accountManager(agentDir: string, callbacks: LoginCallbacks): Promise<StoredPool> {
	let state = readPool(agentDir, KIRO_PROVIDER_ID);
	const now = Date.now();

	if (state.accounts.length === 0 || !callbacks.onSelect) {
		return toStored(await addFlow(state, callbacks));
	}

	const action = await callbacks.onSelect({
		message: `Kiro accounts\n${describe(state, now)}`,
		options: [
			{ id: "add", label: "Add an account" },
			{ id: "remove", label: "Log out of one account" },
			{ id: "logout-all", label: "Log out of every account" },
			{ id: "pin", label: "Pin an account" },
			{ id: "unpin", label: "Clear the pin" },
			{ id: "unblock", label: "Clear a block" },
			{ id: "mode", label: "Scheduling mode" },
			{ id: "migrate", label: "Migration policy" },
		],
	});

	switch (action) {
		case "add":
			state = await addFlow(state, callbacks);
			break;
		case "remove": {
			const target = await pickAccount(state, callbacks, "Log out of which account?");
			if (target) state = removeAccount(state, target);
			break;
		}
		// Full logout. The pin and conversation bindings are dropped with the
		// accounts, or they would point at slots that no longer exist.
		case "logout-all": {
			const confirm = await callbacks.onSelect({
				message: `Log out of all ${state.accounts.length} Kiro account(s)?`,
				options: [
					{ id: "no", label: "Cancel" },
					{ id: "yes", label: "Log out of every account" },
				],
			});
			if (confirm === "yes") {
				state = { ...state, accounts: [], bindings: {} };
				delete (state as { pinned?: string }).pinned;
			}
			break;
		}
		case "pin": {
			const target = await pickAccount(state, callbacks, "Pin which account?");
			if (target) state = pinAccount(state, target);
			break;
		}
		case "unpin":
			state = unpinAccount(state);
			break;
		case "unblock": {
			const target = await pickAccount(state, callbacks, "Clear the block on which account?");
			if (target) {
				state = {
					...state,
					accounts: state.accounts.map((slot) => (slot.name === target ? unblockAccount(slot) : slot)),
				};
			}
			break;
		}
		case "mode": {
			const selected = await callbacks.onSelect({
				message: "Scheduling mode",
				options: [
					{ id: "cache-first", label: "Cache first — hold one account, maximise prompt-cache hits" },
					{ id: "balanced", label: "Balanced — place new conversations by remaining quota" },
					{ id: "spread", label: "Spread — round-robin every request" },
				],
			});
			if (selected) state = { ...state, mode: selected as SchedulingMode };
			break;
		}
		case "migrate": {
			const selected = await callbacks.onSelect({
				message: "When a conversation can no longer use the account holding its warm cache",
				options: [
					{ id: "auto", label: "Auto — move it to another account silently" },
					{ id: "ask", label: "Ask — move it, but report the account it left" },
					{ id: "never", label: "Never — fail the request instead of moving it" },
				],
			});
			if (selected) state = { ...state, migration: selected as MigrationPolicy };
			break;
		}
		default:
			break;
	}

	return toStored(state);
}

export function kiroProviderPackage(deps: KiroProviderDeps = {}): ProviderPackage {
	const caches = new Map<string, AccountUsageCache>();
	const cacheFor = (agentDir: string) => {
		const existing = caches.get(agentDir);
		if (existing) return existing;
		const created = createAccountUsageCache(agentDir, KIRO_PROVIDER_ID);
		caches.set(agentDir, created);
		return created;
	};

	return {
		id: KIRO_PROVIDER_ID,
		label: "Kiro",
		build(context: ProviderBuildContext) {
			return buildKiroProviderConfig(
				context,
				{
					name: "Kiro (Google / GitHub / AWS Builder ID)",
					login: async (callbacks) => accountManager(context.agentDir, callbacks as unknown as LoginCallbacks) as never,
					// The stored credential is a pool of accounts, each refreshed
					// individually at request time, so the wrapper never refreshes.
					refreshToken: async (credentials) => credentials,
					getApiKey: (credentials) => credentials.access,
				},
				{ reportMigration: migrationSink.report, ...deps },
			);
		},
		async accountUsageDetail(context) {
			return readKiroPoolUsage(context, cacheFor(context.agentDir));
		},
		async accountUsage(context) {
			const detail = await readKiroPoolUsage(context, cacheFor(context.agentDir));
			return Object.fromEntries(Object.entries(detail).map(([name, usage]) => [name, headroomOf(usage)]));
		},
	};
}

export { DEFAULT_SCHEDULING_MODE };
