"""FastAPI REST API for the associative memory system.

Production features: auth, rate limiting, structured logging, metrics.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, PlainTextResponse

from amem.config import load_config
from amem.embeddings.factory import create_embedder
from amem.retrieval.orchestrator import MemoryOrchestrator
from amem.utils.auth import AuthManager, get_auth_dependency
from amem.utils.logging import setup_logging, get_logger, metrics, set_request_id
from amem.utils.ratelimit import RateLimiter
from api.models import (
    IngestRequest, IngestConversationRequest, IngestResponse,
    QueryRequest, QueryResponse,
    ExplicitMemoryRequest, ExplicitMemoryUpdateRequest,
    SessionStartRequest, GraphQueryRequest,
    WorkingMemoryAddRequest, StatsResponse,
    MergeEntitiesRequest, AddAliasRequest, RetractFactRequest,
    FeedbackRequest,
)


_orchestrator: MemoryOrchestrator | None = None
logger = get_logger("amem.api")


def create_app(config_path: str | None = None) -> FastAPI:
    config = load_config(config_path)
    setup_logging()

    # Auth and rate limiting
    auth = AuthManager()
    rate_limiter = RateLimiter(rate=10.0, capacity=50)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _orchestrator
        logger.info("Starting amem server")
        embedder = create_embedder(config.embedding if config.embedding.provider != "auto" else config.ollama)
        _orchestrator = MemoryOrchestrator(embedder, config)
        _orchestrator.init_db()
        _orchestrator.load()
        logger.info("Server ready", extra={"data": _orchestrator.stats()})
        yield
        logger.info("Shutting down")
        _orchestrator.save()
        _orchestrator.close()
        if hasattr(embedder, 'close'):
            await embedder.close()

    app = FastAPI(
        title="amem API",
        version="0.3.0",
        description="Five-layer associative memory for AI agents",
        lifespan=lifespan,
    )

    # Middleware: request ID + logging
    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        req_id = str(uuid.uuid4())[:8]
        set_request_id(req_id)
        response = await call_next(request)
        return response

    # Auth dependency (only enforced if API keys are configured)
    auth_dep = get_auth_dependency(auth)

    def orch() -> MemoryOrchestrator:
        if _orchestrator is None:
            raise HTTPException(503, "System not initialized")
        return _orchestrator

    # --- Ingest ---

    @app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(auth_dep)])
    async def ingest(req: IngestRequest, request: Request):
        rate_limiter.check(request)
        metrics.increment("ingest_requests")
        result = await orch().ingest(
            text=req.text,
            conversation_id=req.conversation_id,
            speaker=req.speaker,
            timestamp=req.timestamp,
        )
        orch().save()
        return IngestResponse(**{k: v for k, v in result.items() if k in IngestResponse.model_fields})

    @app.post("/ingest/conversation", response_model=IngestResponse)
    async def ingest_conversation(req: IngestConversationRequest):
        result = await orch().ingest_conversation(
            messages=req.messages,
            conversation_id=req.conversation_id,
        )
        orch().save()
        return IngestResponse(**{k: v for k, v in result.items() if k in IngestResponse.model_fields})

    # --- Query ---

    @app.post("/query", response_model=QueryResponse)
    async def query(req: QueryRequest):
        ctx = await orch().query(
            query_text=req.query,
            top_k=req.top_k,
            token_budget=req.token_budget,
        )
        orch().save()
        return QueryResponse(
            context_text=ctx.to_injection_text(profile=orch().behavioral),
            episodic_chunks=ctx.episodic_chunks,
            semantic_facts=ctx.semantic_facts,
            behavioral_priors=ctx.behavioral_priors,
            explicit_entries=ctx.explicit_entries,
            working_context=ctx.working_context,
            contradictions=ctx.contradictions,
            tokens_estimate=ctx.total_tokens_estimate,
            budget_allocation=ctx.budget_allocation,
        )

    # --- Session ---

    @app.post("/session/start")
    async def session_start(req: SessionStartRequest):
        orch().start_session(req.session_id)
        return {"session_id": orch().working.session_id}

    @app.post("/session/end")
    async def session_end():
        sid = await orch().end_session()
        orch().save()
        return {"session_id": sid, "flushed": sid is not None}

    @app.post("/session/add")
    async def session_add(req: WorkingMemoryAddRequest):
        entry_id = orch().working.add(req.entry_type, req.content)
        return {"entry_id": entry_id}

    @app.get("/session/context")
    async def session_context():
        return orch().working.get_context()

    # --- Explicit Memory ---

    @app.get("/explicit")
    async def list_explicit():
        return [e.to_dict() for e in orch().explicit.list_all()]

    @app.post("/explicit")
    async def add_explicit(req: ExplicitMemoryRequest):
        entry = orch().explicit.set(
            req.key, req.value,
            entry_type=req.entry_type,
            priority=req.priority,
        )
        orch().save()
        return entry.to_dict()

    @app.put("/explicit/{key}")
    async def update_explicit(key: str, req: ExplicitMemoryUpdateRequest):
        existing = orch().explicit.get(key)
        if existing is None:
            raise HTTPException(404, f"Key not found: {key}")
        entry = orch().explicit.set(
            key,
            req.value,
            entry_type=req.entry_type or existing.entry_type,
            priority=req.priority if req.priority is not None else existing.priority,
        )
        orch().save()
        return entry.to_dict()

    @app.delete("/explicit/{key}")
    async def delete_explicit(key: str):
        if not orch().explicit.delete(key):
            raise HTTPException(404, f"Key not found: {key}")
        orch().save()
        return {"deleted": key}

    # --- Semantic Graph ---

    @app.post("/graph/query")
    async def graph_query(req: GraphQueryRequest):
        facts = orch().semantic.query(req.entities, max_depth=req.max_depth)
        return {"facts": facts}

    @app.get("/graph/entities")
    async def graph_entities():
        return {"entities": orch().semantic.get_entities()}

    @app.post("/graph/merge")
    async def merge_entities(req: MergeEntitiesRequest):
        success = orch().merge_entities(req.name_a, req.name_b)
        if not success:
            raise HTTPException(404, "One or both entities not found")
        orch().save()
        return {"merged": True, "canonical": req.name_a, "alias": req.name_b}

    @app.post("/graph/alias")
    async def add_alias(req: AddAliasRequest):
        orch().add_entity_alias(req.canonical_name, req.alias)
        orch().save()
        return {"canonical": req.canonical_name, "alias": req.alias}

    @app.post("/graph/retract")
    async def retract_fact(req: RetractFactRequest):
        success = orch().retract_fact(req.subject, req.predicate, req.object)
        orch().save()
        return {"retracted": success}

    @app.get("/graph/contradictions")
    async def get_contradictions():
        return {"contradictions": orch().semantic.get_contradictions()}

    @app.get("/graph/contradictions/unresolved")
    async def get_unresolved():
        return {"contradictions": orch().semantic.get_unresolved_contradictions()}

    # --- Profile ---

    @app.get("/profile")
    async def get_profile():
        return {
            "priors": orch().behavioral.get_priors(),
            "summary": orch().behavioral.get_summary(),
        }

    @app.post("/profile/feedback")
    async def submit_feedback(req: FeedbackRequest):
        orch().behavioral.update_from_feedback(req.dimension, req.value)
        orch().save()
        return {"dimension": req.dimension, "updated_value": req.value}

    # --- Admin ---

    @app.post("/admin/decay")
    async def run_decay():
        orch().decay_pass()
        orch().save()
        return {"status": "decay pass complete"}

    @app.get("/stats", response_model=StatsResponse)
    async def stats():
        return StatsResponse(**orch().stats())

    # --- Dashboard ---

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        dashboard_path = Path(__file__).parent.parent / "dashboard" / "index.html"
        if dashboard_path.exists():
            return HTMLResponse(content=dashboard_path.read_text())
        return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)

    # --- Health & Metrics ---

    @app.get("/health")
    async def health():
        return {"status": "ok", "auth_enabled": auth.enabled, "stats": orch().stats()}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics():
        """Prometheus-compatible metrics endpoint."""
        s = orch().stats()
        metrics.gauge("episodic_chunks", s.get("episodic", {}).get("count", 0))
        metrics.gauge("semantic_entities", s.get("semantic", {}).get("entities", 0))
        metrics.gauge("explicit_entries", s.get("explicit", {}).get("count", 0))
        return metrics.to_prometheus()

    return app
