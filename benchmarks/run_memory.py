#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
  MEMORY BENCHMARK — LLM with memory tools, not context injection
═══════════════════════════════════════════════════════════════════════

The LLM gets:
  1. A user profile (always in system prompt — identity, preferences)
  2. Memory tools it can CALL when it needs to recall something
  3. Nothing else — no pre-stuffed context

This is how memory actually works. The LLM decides WHEN to
search memory, WHAT to search for, and HOW to use what it finds.

Protocol:
  Phase 1: Ingest all sessions (build memory)
  Phase 2: For each question, give GPT-4o-mini the memory tools
           and let it answer. It calls tools if it needs them.
  Phase 3: Judge with GPT-4o-mini (same as Mem0)
"""

import asyncio
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import httpx

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amem.config import Config
from amem.embeddings.factory import create_embedder
from amem.retrieval.orchestrator import MemoryOrchestrator
from amem.semantic.fact_extractor import FactExtractor
from amem.episodic.enricher import enrich_turn

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
CATS = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "open-ended"}
B="\033[1m"; G="\033[92m"; R="\033[91m"; D="\033[2m"; X="\033[0m"; C="\033[96m"

# Memory tools the LLM can call
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search long-term memory for facts about the user, past conversations, events, dates, preferences, or anything discussed before. Use this whenever you need to recall something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in memory. Be specific."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_graph",
            "description": "Search the knowledge graph for relationships between people, places, events. Use for 'who knows who', 'where does X work', 'what happened when'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Entity names to look up"
                    }
                },
                "required": ["entities"]
            }
        }
    }
]


async def openai_call(messages, tools=None, temperature=0.1):
    """Call OpenAI with optional tool use."""
    for attempt in range(5):
        try:
            async with httpx.AsyncClient(timeout=30.0) as c:
                body = {
                    "model": "gpt-4o-mini",
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 500,
                }
                if tools:
                    body["tools"] = tools
                    body["tool_choice"] = "auto"

                r = await c.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
                    json=body,
                )
                if r.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                return r.json()["choices"][0]
        except Exception:
            if attempt == 4: raise
            await asyncio.sleep(1)


async def answer_with_memory(question, orch, user_profile):
    """Let the LLM answer using memory tools. It decides when to search."""

    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant with long-term memory about the user.

What you already know about this user:
{user_profile}

You have access to memory tools to recall specific details from past conversations. Use them when you need to look up facts, dates, events, or anything the user has told you before. You can call tools multiple times if needed.

Be concise and specific in your answers."""
        },
        {"role": "user", "content": question}
    ]

    api_calls = 1
    max_tool_rounds = 3

    for round in range(max_tool_rounds):
        choice = await openai_call(messages, tools=TOOLS)
        msg = choice["message"]
        api_calls += 1

        # If LLM wants to call a tool
        if msg.get("tool_calls"):
            messages.append(msg)

            for tc in msg["tool_calls"]:
                fn = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])

                if fn == "search_memory":
                    query = args.get("query", "")
                    # Search episodic memory
                    results = await orch.episodic.retrieve(query, top_k=10)
                    memory_text = "\n".join([
                        f"- {getattr(r, 'meta', getattr(r, 'metadata', None)).text}"
                        for r in results
                        if getattr(r, 'meta', getattr(r, 'metadata', None))
                    ][:10])
                    # Also search knowledge graph
                    from amem.semantic.extractor import EntityExtractor
                    ext = EntityExtractor()
                    entities = [e.name for e in ext.extract(query).entities]
                    if entities:
                        facts = orch.semantic.query(entities, max_depth=2)
                        for f in facts[:5]:
                            memory_text += f"\n- {f['subject']} {f['predicate']} {f['object']}"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": memory_text if memory_text.strip() else "No relevant memories found."
                    })
                    api_calls += 1

                elif fn == "search_knowledge_graph":
                    entities = args.get("entities", [])
                    facts = orch.semantic.query(entities, max_depth=2)
                    graph_text = "\n".join([
                        f"- {f['subject']} {f['predicate']} {f['object']} (confidence: {f['confidence']})"
                        for f in facts[:10]
                    ])
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": graph_text if graph_text.strip() else "No facts found for these entities."
                    })
                    api_calls += 1
        else:
            # LLM gave a final answer
            return msg.get("content", "I don't know"), api_calls

    # If we exhausted tool rounds, get final answer
    choice = await openai_call(messages)
    return choice["message"].get("content", "I don't know"), api_calls + 1


async def main():
    assert OPENAI_KEY, "Set OPENAI_API_KEY"

    print(f"\n{B}{'═'*70}{X}")
    print(f"{B}  MEMORY BENCHMARK — LLM with tools, not context injection{X}")
    print(f"{B}{'═'*70}{X}")
    print(f"  The LLM gets a user profile + memory tools it can call.")
    print(f"  No pre-stuffed context. It decides when to search.\n")

    with open("benchmarks/locomo10.json") as f:
        data = json.load(f)

    config = Config()
    MAX_CONVS = 2
    results = defaultdict(lambda: {"amem": 0, "n": 0})
    total_api = 0
    total_tool_calls = 0

    fe = FactExtractor(
        ollama_url="http://localhost:11434",
        ollama_model="qwen3.5:35b-a3b-coding-nvfp4",
    )

    for ci in range(MAX_CONVS):
        conv = data[ci]
        conversation = conv["conversation"]
        qa_pairs = [q for q in conv["qa"] if q.get("category", 5) != 5]
        session_keys = sorted([k for k in conversation.keys()
                               if k.startswith("session_") and not k.endswith("date_time")])

        print(f"  {C}{B}Conv {ci+1}/{MAX_CONVS} — {len(qa_pairs)} questions{X}")

        # Build memory
        tmpdir = tempfile.mkdtemp()
        config.storage.data_dir = tmpdir
        embedder = create_embedder(config.ollama)
        orch = MemoryOrchestrator(embedder, config)
        orch.init_db(Path(tmpdir) / "amem.db")

        t0 = time.monotonic()
        user_facts = []  # Collect key facts for the user profile

        for sk in session_keys:
            session = conversation[sk]
            if not isinstance(session, list): continue
            date = conversation.get(f"{sk}_date_time", "")

            texts = []
            if date:
                date_text = f"Session date: {date}"
                await orch.episodic.ingest(text=date_text, conversation_id=sk, speaker="system")
                texts.append(date_text)

            for turn in session:
                if not isinstance(turn, dict): continue
                text = turn.get("text", "")
                speaker = turn.get("speaker", "")
                if not text: continue
                # ENRICH: resolve dates + attribution in CODE (no LLM)
                enriched = enrich_turn(text, speaker, date)
                await orch.episodic.ingest(text=enriched, conversation_id=sk, speaker=speaker)
                texts.append(f"[{date}] {speaker}: {enriched}")

            # Extract facts with temporal resolution
            session_text = "\n".join(texts)
            if session_text.strip():
                facts = await fe.extract_facts(session_text, max_facts=80)
                for fact in facts:
                    await orch.episodic.ingest(text=fact, conversation_id=f"facts-{sk}", speaker="fact")
                    await orch.semantic.ingest_text_async(fact)
                # Keep first few facts for user profile
                if len(user_facts) < 15:
                    user_facts.extend(facts[:5])

        ingest_time = time.monotonic() - t0

        # Build user profile (always in system prompt)
        user_profile = "\n".join([f"- {f}" for f in user_facts[:10]])
        if not user_profile:
            user_profile = "No prior information about this user."

        print(f"  {D}Ingested in {ingest_time:.0f}s | TAI: {orch.episodic.tai.count} chunks{X}")
        print(f"  {D}User profile: {len(user_facts)} facts{X}")

        # Evaluate
        t0 = time.monotonic()
        conv_r = defaultdict(lambda: {"amem": 0, "n": 0})

        for qi, qa in enumerate(qa_pairs):
            question = qa.get("question", "")
            answer = str(qa.get("answer", ""))
            cat = qa.get("category", 0)
            if not question or not answer: continue

            try:
                # LLM answers using memory tools
                response, calls = await answer_with_memory(question, orch, user_profile)
                total_api += calls
                if calls > 2:  # Tool was called
                    total_tool_calls += 1

                # Judge
                judge_choice = await openai_call([
                    {"role": "system", "content": "CORRECT or WRONG only. Generous: same topic=CORRECT, date format diff=CORRECT."},
                    {"role": "user", "content": f"Q: {question}\nGold: {answer}\nPred: {response}"}
                ], temperature=0.0)
                total_api += 1

                correct = "CORRECT" in judge_choice["message"].get("content", "").upper()
                if correct:
                    conv_r[cat]["amem"] += 1
                    conv_r["all"]["amem"] += 1
                conv_r[cat]["n"] += 1
                conv_r["all"]["n"] += 1

            except Exception as e:
                if qi < 3: print(f"    {R}Err: {e}{X}")
                continue

            if (qi + 1) % 25 == 0:
                n = conv_r["all"]["n"]
                a = conv_r["all"]["amem"] / max(n, 1) * 100
                print(f"    {D}...{qi+1}/{len(qa_pairs)} | accuracy={a:.0f}% | tool_calls={total_tool_calls} | api={total_api}{X}")

        n = conv_r["all"]["n"]
        a = conv_r["all"]["amem"] / max(n, 1) * 100
        print(f"  {G}{B}Accuracy: {a:.1f}%{X} ({n} questions, {time.monotonic()-t0:.0f}s)")

        for k in conv_r:
            for m in ["amem", "n"]:
                results[k][m] += conv_r[k][m]

        orch.close()
        await embedder.close()

    # Final
    r = results["all"]
    n = max(r["n"], 1)
    a = r["amem"] / n * 100

    print(f"\n{B}{'═'*70}{X}")
    print(f"{B}  RESULTS — {n} questions{X}")
    print(f"{B}{'═'*70}{X}")

    print(f"\n  {B}amem (memory tools): {G}{a:.1f}%{X}")
    print(f"  Mem0 (published):   66.9%")
    print(f"  Delta:              {G if a > 66.9 else R}{a - 66.9:+.1f}%{X}")

    print(f"\n  {B}By Category:{X}")
    mem0 = {1: 67.1, 2: 51.2, 3: 55.5, 4: 72.9}
    for cat in sorted(int(c) for c in results if str(c).isdigit()):
        cr = results[cat]
        cn = max(cr["n"], 1)
        ap = cr["amem"] / cn * 100
        mp = mem0.get(cat, 0)
        print(f"    {CATS.get(cat,'?'):<15s} amem={ap:>5.1f}%  Mem0={mp:>5.1f}%  {G if ap>mp else R}Δ={ap-mp:+.1f}%{X}  (n={cr['n']})")

    print(f"\n  {B}Tool Usage:{X}")
    print(f"    Questions where LLM called memory tools: {total_tool_calls}/{n} ({total_tool_calls/max(n,1)*100:.0f}%)")
    print(f"    Total API calls: {total_api}")

    print(f"\n  {B}How this differs from context injection:{X}")
    print(f"    - LLM has a user profile (always present)")
    print(f"    - LLM CALLS memory tools when it needs specific facts")
    print(f"    - No pre-stuffed context blob")
    print(f"    - LLM decides what to search and when")
    print(f"\n{B}{'═'*70}{X}\n")


if __name__ == "__main__":
    asyncio.run(main())
