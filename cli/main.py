"""CLI interface for the associative memory system."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click

from amem.config import load_config
from amem.embeddings.ollama import OllamaEmbedding
from amem.retrieval.orchestrator import MemoryOrchestrator


def _get_orchestrator(config_path: str | None = None) -> MemoryOrchestrator:
    config = load_config(config_path)
    embedder = OllamaEmbedding(config.ollama)
    orch = MemoryOrchestrator(embedder, config)
    orch.load()
    return orch


def _run(coro):
    """Run an async coroutine."""
    return asyncio.run(coro)


@click.group()
@click.option("--config", "-c", default=None, help="Path to config.yaml")
@click.pass_context
def cli(ctx, config):
    """amem — Five-layer associative memory for AI agents"""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


@cli.command()
@click.argument("text", required=False)
@click.option("--file", "-f", type=click.Path(exists=True), help="Read text from file")
@click.option("--conversation-id", "-cid", default=None, help="Conversation ID")
@click.option("--speaker", "-s", default="", help="Speaker name")
@click.pass_context
def ingest(ctx, text, file, conversation_id, speaker):
    """Ingest text or a file into memory."""
    if file:
        text = Path(file).read_text()
    elif text is None:
        text = click.get_text_stream("stdin").read()

    if not text.strip():
        click.echo("Error: No text provided", err=True)
        sys.exit(1)

    orch = _get_orchestrator(ctx.obj["config_path"])

    async def _do():
        result = await orch.ingest(text, conversation_id=conversation_id, speaker=speaker)
        orch.save()
        return result

    result = _run(_do())
    click.echo(f"Ingested: {result['chunks_stored']} chunks, "
               f"{result['entities_extracted']} entities, "
               f"{result['relations_extracted']} relations")
    if result.get("contradictions"):
        click.echo(f"  Contradictions detected: {len(result['contradictions'])}")


@cli.command()
@click.argument("query_text")
@click.option("--top-k", "-k", default=10, help="Number of results")
@click.option("--budget", "-b", default=None, type=int, help="Token budget")
@click.option("--raw", is_flag=True, help="Output raw JSON instead of formatted text")
@click.pass_context
def query(ctx, query_text, top_k, budget, raw):
    """Retrieve relevant memory context for a query."""
    orch = _get_orchestrator(ctx.obj["config_path"])

    async def _do():
        return await orch.query(query_text, top_k=top_k, token_budget=budget or 4000)

    result = _run(_do())
    orch.save()

    if raw:
        click.echo(json.dumps({
            "episodic": result.episodic_chunks,
            "semantic": result.semantic_facts,
            "behavioral": result.behavioral_priors,
            "explicit": result.explicit_entries,
            "working": result.working_context,
            "contradictions": result.contradictions,
            "budget_allocation": result.budget_allocation,
            "tokens_estimate": result.total_tokens_estimate,
        }, indent=2, default=str))
    else:
        click.echo(result.to_injection_text(profile=orch.behavioral))


@cli.command()
@click.argument("key")
@click.argument("value")
@click.option("--type", "-t", "entry_type", default="fact",
              type=click.Choice(["fact", "preference", "instruction", "context"]))
@click.option("--priority", "-p", default=0, type=int)
@click.pass_context
def remember(ctx, key, value, entry_type, priority):
    """Add an explicit memory entry."""
    orch = _get_orchestrator(ctx.obj["config_path"])
    entry = orch.explicit.set(key, value, entry_type=entry_type, priority=priority)
    orch.save()
    click.echo(f"Remembered: [{entry_type}] {key} = {value}")


@cli.command()
@click.argument("key")
@click.pass_context
def forget(ctx, key):
    """Remove an explicit memory entry."""
    orch = _get_orchestrator(ctx.obj["config_path"])
    if orch.explicit.delete(key):
        orch.save()
        click.echo(f"Forgot: {key}")
    else:
        click.echo(f"Not found: {key}", err=True)


@cli.command()
@click.pass_context
def memories(ctx):
    """List all explicit memory entries."""
    orch = _get_orchestrator(ctx.obj["config_path"])
    entries = orch.explicit.list_all()
    if not entries:
        click.echo("No explicit memories stored.")
        return
    for e in entries:
        click.echo(f"  [{e.entry_type}] {e.key}: {e.value} (priority: {e.priority})")


@cli.command()
@click.argument("entity")
@click.option("--depth", "-d", default=2, type=int, help="Graph traversal depth")
@click.pass_context
def graph(ctx, entity, depth):
    """Show knowledge graph facts around an entity."""
    orch = _get_orchestrator(ctx.obj["config_path"])
    facts = orch.semantic.query([entity], max_depth=depth)
    if not facts:
        click.echo(f"No facts found for entity: {entity}")
        return
    for f in facts:
        status = f.get("status", "active")
        status_tag = f" [{status.upper()}]" if status != "active" else ""
        click.echo(f"  {f['subject']} —[{f['predicate']}]→ {f['object']}  "
                   f"(confidence: {f['confidence']}, mentions: {f.get('mention_count', 1)}){status_tag}")


@cli.command()
@click.argument("name_a")
@click.argument("name_b")
@click.pass_context
def merge(ctx, name_a, name_b):
    """Merge two entities (name_a becomes canonical, name_b becomes alias)."""
    orch = _get_orchestrator(ctx.obj["config_path"])
    success = orch.merge_entities(name_a, name_b)
    if success:
        orch.save()
        click.echo(f"Merged: {name_b} → {name_a}")
    else:
        click.echo("Failed: one or both entities not found", err=True)


@cli.command()
@click.argument("canonical_name")
@click.argument("alias")
@click.pass_context
def alias(ctx, canonical_name, alias):
    """Add an alias for a known entity."""
    orch = _get_orchestrator(ctx.obj["config_path"])
    orch.add_entity_alias(canonical_name, alias)
    orch.save()
    click.echo(f"Alias added: {alias} → {canonical_name}")


@cli.command()
@click.argument("subject")
@click.argument("predicate")
@click.argument("obj")
@click.pass_context
def retract(ctx, subject, predicate, obj):
    """Retract a fact from the knowledge graph."""
    orch = _get_orchestrator(ctx.obj["config_path"])
    success = orch.retract_fact(subject, predicate, obj)
    orch.save()
    if success:
        click.echo(f"Retracted: {subject} —[{predicate}]→ {obj}")
    else:
        click.echo("Fact not found", err=True)


@cli.command()
@click.pass_context
def contradictions(ctx):
    """Show unresolved contradictions in the knowledge graph."""
    orch = _get_orchestrator(ctx.obj["config_path"])
    unresolved = orch.semantic.get_unresolved_contradictions()
    if not unresolved:
        click.echo("No unresolved contradictions.")
        return
    for c in unresolved:
        fa = c.get("fact_a", {})
        fb = c.get("fact_b", {})
        click.echo(f"  CONFLICT ({c.get('contradiction_type', '?')}):")
        click.echo(f"    A: {fa.get('subject')} {fa.get('predicate')} {fa.get('object')}")
        click.echo(f"    B: {fb.get('subject')} {fb.get('predicate')} {fb.get('object')}")


@cli.command()
@click.pass_context
def profile(ctx):
    """Show the current behavioral profile."""
    orch = _get_orchestrator(ctx.obj["config_path"])
    summary = orch.behavioral.get_summary()
    for dim, desc in summary.items():
        click.echo(f"  {dim}: {desc}")


@cli.command()
@click.argument("dimension")
@click.argument("value", type=float)
@click.pass_context
def feedback(ctx, dimension, value):
    """Provide explicit behavioral feedback (dimension: 0.0-1.0)."""
    orch = _get_orchestrator(ctx.obj["config_path"])
    orch.behavioral.update_from_feedback(dimension, value)
    orch.save()
    click.echo(f"Feedback recorded: {dimension} = {value}")


@cli.command()
@click.pass_context
def decay(ctx):
    """Run a decay pass on all memory layers."""
    orch = _get_orchestrator(ctx.obj["config_path"])
    orch.decay_pass()
    orch.save()
    click.echo("Decay pass complete.")


@cli.command()
@click.pass_context
def status(ctx):
    """Show memory system statistics."""
    orch = _get_orchestrator(ctx.obj["config_path"])
    stats = orch.stats()
    click.echo(json.dumps(stats, indent=2, default=str))


@cli.command()
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=8420, type=int, help="Server port")
@click.pass_context
def serve(ctx, host, port):
    """Start the REST API server."""
    import uvicorn
    from api.app import create_app

    app = create_app(ctx.obj["config_path"])
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()
