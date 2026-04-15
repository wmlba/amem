"""Knowledge Assembler — the new injection strategy.

OLD approach (broken):
    Query → retrieve raw chunks → inject as context → LLM answers
    Problem: injects fragments the LLM already has in its conversation

NEW approach:
    Query → assemble CROSS-SESSION KNOWLEDGE the LLM doesn't have:
    1. Who is this user? (explicit: identity, preferences, instructions)
    2. What do I know from past sessions? (semantic: facts, relationships)
    3. What relevant details from history? (episodic: key past exchanges, summarized)
    4. How should I interact? (behavioral: tone, depth)

The current conversation IS the working memory — the LLM already has it.
We provide everything it DOESN'T have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from amem.utils.tokenizer import count_tokens


@dataclass
class KnowledgeContext:
    """Assembled knowledge for injection — cross-session only."""

    # Who is this user?
    user_identity: list[dict] = field(default_factory=list)      # explicit facts
    user_instructions: list[dict] = field(default_factory=list)   # explicit instructions
    user_preferences: list[dict] = field(default_factory=list)    # explicit preferences

    # What do I know from past sessions?
    known_facts: list[dict] = field(default_factory=list)         # semantic graph relations
    known_entities: list[dict] = field(default_factory=list)      # key entities with properties

    # Relevant history (session summaries, not raw chunks)
    past_session_summaries: list[dict] = field(default_factory=list)  # compressed past sessions
    relevant_episodes: list[dict] = field(default_factory=list)       # specific past exchanges when needed

    # How should I interact?
    behavioral_profile: dict = field(default_factory=dict)

    # Metadata
    total_tokens: int = 0
    sessions_recalled: int = 0

    def to_injection_text(self) -> str:
        """Format as structured knowledge block for the LLM system prompt."""
        parts = []

        # User identity + instructions (ALWAYS injected, highest priority)
        identity_items = self.user_identity + self.user_preferences
        if identity_items or self.user_instructions:
            parts.append("## About This User")
            for item in identity_items:
                parts.append(f"- {item['key']}: {item['value']}")
            for item in self.user_instructions:
                parts.append(f"- [instruction] {item['key']}: {item['value']}")

        # Known facts from knowledge graph
        if self.known_facts:
            parts.append("\n## Known Facts (from past conversations)")
            for fact in self.known_facts:
                status = fact.get("status", "active")
                if status == "active":
                    parts.append(f"- {fact['subject']} {fact['predicate']} {fact['object']}")

        # Past session summaries
        if self.past_session_summaries:
            parts.append("\n## Previous Sessions")
            for summary in self.past_session_summaries:
                session_id = summary.get("session_id", "")
                text = summary.get("summary", summary.get("text", ""))
                parts.append(f"- [{session_id}] {text}")

        # Specific relevant episodes (when query needs historical detail)
        if self.relevant_episodes:
            parts.append("\n## Relevant Past Details")
            for ep in self.relevant_episodes:
                parts.append(f"- {ep['text']}")

        # Behavioral hints (compact)
        if self.behavioral_profile:
            profile = self.behavioral_profile
            hints = []
            depth = profile.get("response_depth", {}).get("value", 0.5)
            if depth > 0.7:
                hints.append("prefers detailed technical responses")
            elif depth < 0.3:
                hints.append("prefers brief responses")
            formality = profile.get("formality", {}).get("value", 0.5)
            if formality > 0.7:
                hints.append("formal tone")
            elif formality < 0.3:
                hints.append("casual tone")
            if hints:
                parts.append(f"\n## Communication Style")
                parts.append(f"- {', '.join(hints)}")

        text = "\n".join(parts)
        self.total_tokens = count_tokens(text)
        return text


def assemble_knowledge(
    explicit_entries: list[dict],
    semantic_facts: list[dict],
    episodic_chunks: list[dict],
    behavioral_priors: dict,
    session_summaries: list[dict] | None = None,
    current_session_id: str | None = None,
    token_budget: int = 2000,
) -> KnowledgeContext:
    """Assemble cross-session knowledge for injection.

    Key principle: EXCLUDE anything from the current session.
    The LLM already has the current conversation in its context window.
    We provide what it DOESN'T have.
    """
    ctx = KnowledgeContext()

    # 1. Explicit memory — always included, highest priority
    for entry in explicit_entries:
        etype = entry.get("entry_type", "fact")
        if etype == "instruction":
            ctx.user_instructions.append(entry)
        elif etype == "preference":
            ctx.user_preferences.append(entry)
        else:
            ctx.user_identity.append(entry)

    # 2. Semantic facts — structured knowledge from past sessions
    for fact in semantic_facts:
        if fact.get("status", "active") == "active":
            ctx.known_facts.append(fact)

    # 3. Session summaries — compressed past sessions (not raw chunks)
    if session_summaries:
        for summary in session_summaries:
            # Skip current session
            if current_session_id and summary.get("session_id") == current_session_id:
                continue
            ctx.past_session_summaries.append(summary)
        ctx.sessions_recalled = len(ctx.past_session_summaries)

    # 4. Episodic chunks — include ALL relevant chunks (raw + extracted facts)
    # FIX 3: We now store LLM-extracted facts as searchable episodic chunks.
    # These show up here alongside raw conversation chunks.
    # Skip only current-session RAW conversation (the LLM has it), but
    # INCLUDE extracted facts from any session (speaker="fact").
    for chunk in episodic_chunks:
        conv_id = chunk.get("conversation_id", "")
        speaker = chunk.get("speaker", "")

        # Always include extracted facts (they're high-value knowledge)
        if speaker == "fact":
            ctx.relevant_episodes.append(chunk)
            continue

        # Skip current session's raw conversation (LLM already has it)
        if current_session_id and current_session_id in conv_id:
            continue

        # Include high-relevance past chunks
        if chunk.get("score", 0) > 0.3:
            ctx.relevant_episodes.append(chunk)

    # 5. Behavioral profile
    ctx.behavioral_profile = behavioral_priors

    # Trim to budget — prioritize: explicit > semantic > summaries > episodes
    text = ctx.to_injection_text()
    while ctx.total_tokens > token_budget and ctx.relevant_episodes:
        ctx.relevant_episodes.pop()
        ctx.to_injection_text()
    while ctx.total_tokens > token_budget and ctx.past_session_summaries:
        ctx.past_session_summaries.pop()
        ctx.to_injection_text()
    while ctx.total_tokens > token_budget and ctx.known_facts:
        ctx.known_facts.pop()
        ctx.to_injection_text()

    return ctx
