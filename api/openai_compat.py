"""OpenAI-compatible proxy that adds associative memory to any LLM.

Intercepts /v1/chat/completions requests:
1. Before forwarding: queries memory and injects context as a system message
2. After receiving response: ingests the conversation turn into memory

Drop-in replacement: just change base_url to point at this proxy.

Usage:
    python -m api.openai_compat --target https://api.openai.com/v1 --port 8421
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from amem.config import load_config
from amem.embeddings.ollama import OllamaEmbedding
from amem.retrieval.orchestrator import MemoryOrchestrator


_orchestrator: MemoryOrchestrator | None = None
_target_client: httpx.AsyncClient | None = None
_target_base: str = ""


def create_proxy_app(
    config_path: str | None = None,
    target_url: str = "https://api.openai.com/v1",
) -> FastAPI:
    config = load_config(config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _orchestrator, _target_client, _target_base
        _target_base = target_url.rstrip("/")
        _target_client = httpx.AsyncClient(timeout=120.0)

        embedder = OllamaEmbedding(config.ollama)
        _orchestrator = MemoryOrchestrator(embedder, config)
        _orchestrator.init_db()
        _orchestrator.load()
        yield
        _orchestrator.save()
        _orchestrator.close()
        await _target_client.aclose()
        await embedder.close()

    app = FastAPI(
        title="Associative Memory OpenAI Proxy",
        version="0.1.0",
        lifespan=lifespan,
    )

    def orch() -> MemoryOrchestrator:
        assert _orchestrator is not None
        return _orchestrator

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        user_id = request.headers.get("X-User-ID", "default")

        # 1. Extract the latest user message for memory query
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        # 2. Query memory for relevant context
        memory_context = ""
        if last_user_msg:
            try:
                ctx = await orch().query(last_user_msg, top_k=10)
                memory_context = ctx.to_injection_text(profile=orch().behavioral)
            except Exception:
                pass  # Don't block the request if memory fails

        # 3. Inject memory as a system message prefix
        if memory_context:
            memory_system_msg = {
                "role": "system",
                "content": (
                    f"[Associative Memory Context]\n"
                    f"The following is retrieved from the user's memory. "
                    f"Use it to personalize your response.\n\n"
                    f"{memory_context}"
                ),
            }
            # Prepend to messages (after any existing system message)
            if messages and messages[0].get("role") == "system":
                messages.insert(1, memory_system_msg)
            else:
                messages.insert(0, memory_system_msg)
            body["messages"] = messages

        # 4. Forward to target LLM
        headers = dict(request.headers)
        # Remove host header (will be set by httpx)
        headers.pop("host", None)
        headers.pop("content-length", None)

        stream = body.get("stream", False)

        if stream:
            return await _handle_streaming(body, headers, last_user_msg, orch())
        else:
            return await _handle_non_streaming(body, headers, last_user_msg, orch())

    async def _handle_non_streaming(body, headers, user_msg, orchestrator):
        resp = await _target_client.post(
            f"{_target_base}/chat/completions",
            json=body,
            headers=headers,
        )

        resp_data = resp.json()

        # 5. Ingest the conversation turn into memory
        try:
            if user_msg:
                await orchestrator.ingest(text=user_msg, speaker="user")
            # Also ingest the assistant response
            choices = resp_data.get("choices", [])
            if choices:
                assistant_msg = choices[0].get("message", {}).get("content", "")
                if assistant_msg:
                    await orchestrator.ingest(text=assistant_msg, speaker="assistant")
            orchestrator.save()
        except Exception:
            pass

        return Response(
            content=json.dumps(resp_data),
            status_code=resp.status_code,
            media_type="application/json",
        )

    async def _handle_streaming(body, headers, user_msg, orchestrator):
        async def stream_generator():
            full_response = []
            async with _target_client.stream(
                "POST",
                f"{_target_base}/chat/completions",
                json=body,
                headers=headers,
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk
                    # Collect response text for memory ingestion
                    try:
                        for line in chunk.decode().split("\n"):
                            if line.startswith("data: ") and line != "data: [DONE]":
                                data = json.loads(line[6:])
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_response.append(content)
                    except (json.JSONDecodeError, IndexError, KeyError):
                        pass

            # After stream completes, ingest into memory
            try:
                if user_msg:
                    await orchestrator.ingest(text=user_msg, speaker="user")
                response_text = "".join(full_response)
                if response_text:
                    await orchestrator.ingest(text=response_text, speaker="assistant")
                orchestrator.save()
            except Exception:
                pass

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # Pass-through for other OpenAI endpoints
    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def proxy_passthrough(request: Request, path: str):
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)

        body = await request.body()
        resp = await _target_client.request(
            method=request.method,
            url=f"{_target_base}/{path}",
            headers=headers,
            content=body,
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    # Health check
    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "memory_stats": orch().stats() if _orchestrator else {},
            "target": _target_base,
        }

    return app


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="OpenAI-compatible Memory Proxy")
    parser.add_argument("--target", default="https://api.openai.com/v1", help="Target LLM API base URL")
    parser.add_argument("--port", type=int, default=8421, help="Proxy port")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    app = create_proxy_app(config_path=args.config, target_url=args.target)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
