"""MCP (Model Context Protocol) server for associative memory.

Implements the MCP JSON-RPC protocol over stdio, exposing memory
operations as tools that any MCP-compatible client can use.

Protocol: JSON-RPC 2.0 over stdin/stdout
Spec: https://modelcontextprotocol.io

Usage:
    python -m mcp.server [--config config.yaml] [--db path/to/amem.db]
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from amem.config import load_config
from amem.embeddings.ollama import OllamaEmbedding
from amem.retrieval.orchestrator import MemoryOrchestrator

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "amem"
SERVER_VERSION = "0.2.0"

# ─── Tool Definitions ───────────────────────────────────────────────

TOOLS = [
    {
        "name": "memory_ingest",
        "description": (
            "Ingest text into associative memory. Stores in episodic (vector) memory, "
            "extracts entities/relations into knowledge graph, and updates behavioral profile. "
            "Use after every user message to build memory over time."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to ingest into memory"},
                "conversation_id": {"type": "string", "description": "ID to group related messages"},
                "speaker": {"type": "string", "description": "Who said this (e.g. 'user', 'assistant')"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "memory_query",
        "description": (
            "Retrieve relevant memory context for a query. Returns episodic chunks, "
            "knowledge graph facts, behavioral priors, and explicit memories — all "
            "ranked by relevance, recency, and confidence. Use at the start of each "
            "response to inject personalized context."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The query to find relevant memories for"},
                "top_k": {"type": "integer", "description": "Max episodic chunks to return", "default": 10},
                "token_budget": {"type": "integer", "description": "Max tokens for context", "default": 4000},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_remember",
        "description": (
            "Store an explicit memory that the user wants remembered. These have highest "
            "priority and never decay. Use when the user says 'remember that...' or "
            "'always do X' or 'my name is...'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Short identifier for this memory"},
                "value": {"type": "string", "description": "The value to remember"},
                "entry_type": {
                    "type": "string",
                    "enum": ["fact", "preference", "instruction", "context"],
                    "default": "fact",
                },
                "priority": {"type": "integer", "description": "Priority (higher = more important)", "default": 0},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "memory_forget",
        "description": "Remove an explicit memory by key. Use when the user says 'forget that...' or 'don't remember X'.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "The key of the memory to forget"},
            },
            "required": ["key"],
        },
    },
    {
        "name": "memory_list",
        "description": "List all explicit memories. Shows what the system explicitly remembers about the user.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "memory_graph",
        "description": (
            "Query the knowledge graph for facts about specific entities. "
            "Returns structured relations like 'Alice works_on ML Pipeline'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Entity names to look up",
                },
                "max_depth": {"type": "integer", "description": "Graph traversal depth", "default": 2},
            },
            "required": ["entities"],
        },
    },
    {
        "name": "memory_retract",
        "description": (
            "Retract a fact from the knowledge graph. Use when the user corrects "
            "something: 'Actually, I don't work at OCI anymore'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "object": {"type": "string"},
            },
            "required": ["subject", "predicate", "object"],
        },
    },
    {
        "name": "memory_merge_entities",
        "description": (
            "Merge two entities that refer to the same thing. "
            "E.g., merge 'GB10' and 'Blackwell workstation' into one entity."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "canonical": {"type": "string", "description": "The name to keep as canonical"},
                "alias": {"type": "string", "description": "The name to merge as an alias"},
            },
            "required": ["canonical", "alias"],
        },
    },
    {
        "name": "memory_stats",
        "description": "Get statistics about the memory system: chunk count, entity count, etc.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ─── MCP Server ──────────────────────────────────────────────────────

class MCPServer:
    """MCP server exposing associative memory tools over stdio JSON-RPC."""

    def __init__(self, config_path: str | None = None, db_path: str | None = None):
        self._config = load_config(config_path)
        self._db_path = db_path
        self._orch: MemoryOrchestrator | None = None
        self._request_id = 0

    async def initialize(self):
        """Initialize the orchestrator and memory layers."""
        embedder = OllamaEmbedding(self._config.ollama)
        self._orch = MemoryOrchestrator(embedder, self._config)
        if self._db_path:
            self._orch.init_db(Path(self._db_path))
        else:
            self._orch.init_db()
        self._orch.load()

    async def shutdown(self):
        if self._orch:
            self._orch.save()
            self._orch.close()

    @property
    def orch(self) -> MemoryOrchestrator:
        assert self._orch is not None, "Server not initialized"
        return self._orch

    # ─── JSON-RPC Dispatch ───────────────────────────────────────────

    async def handle_message(self, message: dict) -> dict | None:
        """Handle a single JSON-RPC message. Returns response or None for notifications."""
        method = message.get("method", "")
        params = message.get("params", {})
        msg_id = message.get("id")

        # Notifications (no id) don't get responses
        if msg_id is None and method.startswith("notifications/"):
            return None

        try:
            if method == "initialize":
                result = await self._handle_initialize(params)
            elif method == "tools/list":
                result = self._handle_tools_list()
            elif method == "tools/call":
                result = await self._handle_tool_call(params)
            elif method == "ping":
                result = {}
            else:
                return self._error_response(msg_id, -32601, f"Method not found: {method}")

            return {"jsonrpc": "2.0", "id": msg_id, "result": result}
        except Exception as e:
            return self._error_response(msg_id, -32603, str(e))

    async def _handle_initialize(self, params: dict) -> dict:
        await self.initialize()
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
        }

    def _handle_tools_list(self) -> dict:
        return {"tools": TOOLS}

    async def _handle_tool_call(self, params: dict) -> dict:
        tool_name = params.get("name", "")
        args = params.get("arguments", {})

        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            raise ValueError(f"Unknown tool: {tool_name}")

        result = await handler(args)
        return {
            "content": [{"type": "text", "text": json.dumps(result, default=str)}],
        }

    def _error_response(self, msg_id: Any, code: int, message: str) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": code, "message": message},
        }

    # ─── Tool Implementations ────────────────────────────────────────

    async def _tool_memory_ingest(self, args: dict) -> dict:
        result = await self.orch.ingest(
            text=args["text"],
            conversation_id=args.get("conversation_id"),
            speaker=args.get("speaker", ""),
        )
        self.orch.save()
        return result

    async def _tool_memory_query(self, args: dict) -> dict:
        ctx = await self.orch.query(
            query_text=args["query"],
            top_k=args.get("top_k", 10),
            token_budget=args.get("token_budget", 4000),
        )
        self.orch.save()
        return {
            "context_text": ctx.to_injection_text(profile=self.orch.behavioral),
            "episodic_count": len(ctx.episodic_chunks),
            "semantic_facts_count": len(ctx.semantic_facts),
            "explicit_count": len(ctx.explicit_entries),
            "tokens_estimate": ctx.total_tokens_estimate,
        }

    async def _tool_memory_remember(self, args: dict) -> dict:
        entry = self.orch.explicit.set(
            key=args["key"],
            value=args["value"],
            entry_type=args.get("entry_type", "fact"),
            priority=args.get("priority", 0),
        )
        self.orch.save()
        return {"remembered": True, "key": args["key"], "value": args["value"]}

    async def _tool_memory_forget(self, args: dict) -> dict:
        deleted = self.orch.explicit.delete(args["key"])
        self.orch.save()
        return {"forgotten": deleted, "key": args["key"]}

    async def _tool_memory_list(self, args: dict) -> dict:
        entries = self.orch.explicit.get_all_for_context()
        return {"memories": entries, "count": len(entries)}

    async def _tool_memory_graph(self, args: dict) -> dict:
        facts = self.orch.semantic.query(
            args["entities"],
            max_depth=args.get("max_depth", 2),
        )
        return {"facts": facts, "count": len(facts)}

    async def _tool_memory_retract(self, args: dict) -> dict:
        success = self.orch.retract_fact(args["subject"], args["predicate"], args["object"])
        self.orch.save()
        return {"retracted": success}

    async def _tool_memory_merge_entities(self, args: dict) -> dict:
        success = self.orch.merge_entities(args["canonical"], args["alias"])
        self.orch.save()
        return {"merged": success, "canonical": args["canonical"], "alias": args["alias"]}

    async def _tool_memory_stats(self, args: dict) -> dict:
        return self.orch.stats()

    # ─── Stdio Transport ─────────────────────────────────────────────

    async def run_stdio(self):
        """Run the MCP server over stdin/stdout."""
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        # Write to stdout
        writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, asyncio.get_event_loop())

        try:
            while True:
                # Read JSON-RPC message (Content-Length header + body)
                line = await reader.readline()
                if not line:
                    break

                line_str = line.decode("utf-8").strip()

                # Handle Content-Length header protocol
                if line_str.startswith("Content-Length:"):
                    content_length = int(line_str.split(":")[1].strip())
                    await reader.readline()  # empty separator line
                    body = await reader.readexactly(content_length)
                    message = json.loads(body.decode("utf-8"))
                else:
                    # Try parsing as raw JSON (some clients skip headers)
                    try:
                        message = json.loads(line_str)
                    except json.JSONDecodeError:
                        continue

                response = await self.handle_message(message)

                if response is not None:
                    response_bytes = json.dumps(response).encode("utf-8")
                    header = f"Content-Length: {len(response_bytes)}\r\n\r\n"
                    writer.write(header.encode("utf-8"))
                    writer.write(response_bytes)
                    await writer.drain()
        except (asyncio.CancelledError, EOFError, ConnectionResetError):
            pass
        finally:
            await self.shutdown()


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="amem MCP Server")
    parser.add_argument("--config", "-c", default=None, help="Path to config.yaml")
    parser.add_argument("--db", default=None, help="Path to amem.db")
    args = parser.parse_args()

    server = MCPServer(config_path=args.config, db_path=args.db)
    await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
