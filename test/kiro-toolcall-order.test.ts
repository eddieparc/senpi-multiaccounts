import type { Api, Context, Model } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DebugLogger } from "../src/providers/kiro/vendor/debug-logger.js";
import { crc32 } from "../src/providers/kiro/vendor/eventstream.js";
import { createKiroStream } from "../src/providers/kiro/vendor/kiro.js";

const originalFetch = globalThis.fetch;
const encoder = new TextEncoder();

afterEach(() => {
	globalThis.fetch = originalFetch;
});

function bytes(...parts: Uint8Array[]): Uint8Array {
	const output = new Uint8Array(parts.reduce((length, part) => length + part.length, 0));
	let offset = 0;
	for (const part of parts) {
		output.set(part, offset);
		offset += part.length;
	}
	return output;
}

function stringHeader(name: string, value: string): Uint8Array {
	const encodedName = encoder.encode(name);
	const encodedValue = encoder.encode(value);
	const output = new Uint8Array(1 + encodedName.length + 1 + 2 + encodedValue.length);
	const view = new DataView(output.buffer);
	output[0] = encodedName.length;
	output.set(encodedName, 1);
	output[1 + encodedName.length] = 7;
	view.setUint16(2 + encodedName.length, encodedValue.length, false);
	output.set(encodedValue, 4 + encodedName.length);
	return output;
}

function eventFrame(eventType: string, payload: Record<string, unknown>): Uint8Array {
	const headers = bytes(stringHeader(":message-type", "event"), stringHeader(":event-type", eventType));
	const body = encoder.encode(JSON.stringify(payload));
	const frame = new Uint8Array(16 + headers.length + body.length);
	const view = new DataView(frame.buffer);
	view.setUint32(0, frame.length, false);
	view.setUint32(4, headers.length, false);
	view.setUint32(8, crc32(frame.subarray(0, 8)), false);
	frame.set(headers, 12);
	frame.set(body, 12 + headers.length);
	view.setUint32(frame.length - 4, crc32(frame.subarray(0, frame.length - 4)), false);
	return frame;
}

function model(): Model<Api> {
	return {
		id: "claude-opus-5",
		name: "Claude Opus 5",
		api: "kiro-codewhisperer",
		provider: "kiro",
		baseUrl: "https://example.invalid",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 1_000_000,
		maxTokens: 32_000,
	};
}

async function collectEvents(frames: Uint8Array[]): Promise<Record<string, unknown>[]> {
	globalThis.fetch = vi.fn(async () => {
		const body = new ReadableStream<Uint8Array>({
			start(controller) {
				for (const frame of frames) controller.enqueue(frame);
				controller.close();
			},
		});
		return new Response(body, { status: 200 });
	});
	const logger = new DebugLogger({ extensionRoot: "/tmp/senpi-accounts-test", debug: false });
	const stream = createKiroStream(
		{
			providerId: "kiro",
			upstreamUrl: "https://example.invalid/generate",
			endpoint: "codewhisperer",
			apiKey: "managed",
			requestTimeoutMs: 1_000,
			headers: {},
		},
		{},
		logger,
	)(model(), { messages: [{ role: "user", content: "go", timestamp: Date.now() }] } as Context, {
		apiKey: "access-token",
	});
	const events: Record<string, unknown>[] = [];
	for await (const event of stream as unknown as AsyncIterable<Record<string, unknown>>) events.push(event);
	return events;
}

describe("Kiro native tool-call event order", () => {
	it("closes each tool call before opening the next one", async () => {
		const events = await collectEvents([
			eventFrame("assistantResponseEvent", { content: "running two tools", messageId: "response-1" }),
			eventFrame("toolUseEvent", { toolUseId: "toolu_1", name: "Bash", input: '{"command":"echo a"}' }),
			eventFrame("toolUseEvent", { toolUseId: "toolu_2", name: "Bash", input: '{"command":"echo b"}' }),
		]);

		const lifecycle = events
			.filter((event) => typeof event.type === "string" && String(event.type).startsWith("toolcall_"))
			.map((event) => `${String(event.type)}@${String(event.contentIndex)}`);

		expect(lifecycle).toEqual([
			"toolcall_start@1",
			"toolcall_delta@1",
			"toolcall_end@1",
			"toolcall_start@2",
			"toolcall_delta@2",
			"toolcall_end@2",
		]);
	});

	it("keeps a single tool call intact", async () => {
		const events = await collectEvents([
			eventFrame("toolUseEvent", { toolUseId: "toolu_only", name: "Bash", input: '{"command":"echo solo"}' }),
		]);

		const lifecycle = events
			.filter((event) => typeof event.type === "string" && String(event.type).startsWith("toolcall_"))
			.map((event) => String(event.type));

		expect(lifecycle).toEqual(["toolcall_start", "toolcall_delta", "toolcall_end"]);
		const end = events.find((event) => event.type === "toolcall_end");
		expect(end?.toolCall).toMatchObject({ id: "toolu_only", name: "Bash", arguments: { command: "echo solo" } });
	});
});

describe("Kiro request built from a transcript context", () => {
	it("sends tools and the system prompt that arrive as system messages", async () => {
		let sent: Record<string, unknown> | undefined;
		globalThis.fetch = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
			sent = JSON.parse(String(init?.body));
			return new Response(new ReadableStream<Uint8Array>({ start: (controller) => controller.close() }), { status: 200 });
		});
		const logger = new DebugLogger({ extensionRoot: "/tmp/senpi-accounts-test", debug: false });
		const context = {
			messages: [
				{
					role: "system",
					content: "transcript system prompt",
					toolsAdded: [
						{ name: "read", description: "Read a file", parameters: { type: "object", properties: { path: { type: "string" } } } },
						{ name: "gone", description: "Removed later", parameters: { type: "object", properties: {} } },
					],
					timestamp: 0,
				},
				{ role: "system", content: "", toolsRemoved: [{ name: "gone" }], timestamp: 1 },
				{ role: "user", content: "go", timestamp: 2 },
			],
		} as unknown as Context;
		const stream = createKiroStream(
			{
				providerId: "kiro",
				upstreamUrl: "https://example.invalid/generate",
				endpoint: "codewhisperer",
				apiKey: "managed",
				requestTimeoutMs: 1_000,
				headers: {},
			},
			{},
			logger,
		)(model(), context, { apiKey: "access-token" });
		for await (const _event of stream as unknown as AsyncIterable<unknown>) {
			// drain
		}

		const state = sent?.conversationState as {
			currentMessage: { userInputMessage: { content: string; userInputMessageContext?: { tools?: { toolSpecification: { name: string } }[] } } };
		};
		const tools = state.currentMessage.userInputMessage.userInputMessageContext?.tools ?? [];
		expect(tools.map((tool) => tool.toolSpecification.name)).toEqual(["read"]);
		expect(JSON.stringify(sent)).toContain("transcript system prompt");
	});
});
