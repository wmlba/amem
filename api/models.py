"""Pydantic request/response models for the REST API."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# --- Requests ---

class IngestRequest(BaseModel):
    text: str
    conversation_id: Optional[str] = None
    speaker: str = ""
    timestamp: Optional[datetime] = None


class IngestConversationRequest(BaseModel):
    messages: List[Dict[str, str]]
    conversation_id: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    token_budget: int = 4000
    raw: bool = False


class ExplicitMemoryRequest(BaseModel):
    key: str
    value: Any
    entry_type: str = "fact"
    priority: int = 0


class ExplicitMemoryUpdateRequest(BaseModel):
    value: Any
    entry_type: Optional[str] = None
    priority: Optional[int] = None


class SessionStartRequest(BaseModel):
    session_id: Optional[str] = None


class GraphQueryRequest(BaseModel):
    entities: List[str]
    max_depth: int = 2


class WorkingMemoryAddRequest(BaseModel):
    entry_type: str = "note"
    content: str


class MergeEntitiesRequest(BaseModel):
    name_a: str
    name_b: str


class AddAliasRequest(BaseModel):
    canonical_name: str
    alias: str


class RetractFactRequest(BaseModel):
    subject: str
    predicate: str
    object: str


class FeedbackRequest(BaseModel):
    dimension: str
    value: float


# --- Responses ---

class IngestResponse(BaseModel):
    chunks_stored: int
    entities_extracted: int
    relations_extracted: int


class QueryResponse(BaseModel):
    context_text: str
    episodic_chunks: List[Dict] = []
    semantic_facts: List[Dict] = []
    behavioral_priors: Dict = {}
    explicit_entries: List[Dict] = []
    working_context: Dict = {}
    contradictions: List[Dict] = []
    tokens_estimate: int = 0
    budget_allocation: Dict = {}


class StatsResponse(BaseModel):
    episodic: Dict = {}
    semantic: Dict = {}
    explicit: Dict = {}
    behavioral: Dict = {}
    working: Dict = {}
