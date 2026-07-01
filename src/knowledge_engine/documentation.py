"""Automatic transcript documentation.

Every harboured transcript gets a human-readable markdown document that records its
provenance metadata and its exact chunk breakdown. These documents are the durable,
reviewable record of what entered the knowledge engine and how it was split — they
sit alongside the raw transcript under the data directory and (for a GitHub project)
can be browsed directly.

Pure/deterministic: no network, no LLM, no fallbacks.
"""

from __future__ import annotations

from .chunking import Chunk


def render_document(
    *,
    transcript_id: str,
    domain: str,
    entity_name: str,
    source_kind: str,
    source_id: str,
    source_ref: str | None,
    sha256: str,
    char_count: int,
    ingested_at: str,
    chunks: list[Chunk],
) -> str:
    """Render the canonical markdown document for a single transcript."""
    lines: list[str] = [
        f"# Transcript — {entity_name}",
        "",
        "## Metadata",
        "",
        f"- **Transcript ID:** `{transcript_id}`",
        f"- **Domain:** {domain}",
        f"- **Entity:** {entity_name}",
        f"- **Source kind:** {source_kind}",
        f"- **Source id:** {source_id}",
    ]
    if source_ref:
        lines.append(f"- **Source ref:** {source_ref}")
    lines.extend(
        [
            f"- **SHA-256:** `{sha256}`",
            f"- **Ingested at:** {ingested_at}",
            f"- **Characters:** {char_count}",
            f"- **Chunks:** {len(chunks)}",
            "",
            "## Chunks",
            "",
        ]
    )

    if not chunks:
        lines.append("_No chunks were produced (empty transcript)._")
        return "\n".join(lines) + "\n"

    for chunk in chunks:
        lines.extend(
            [
                f"### Chunk {chunk.index}",
                "",
                f"_Characters {chunk.char_start}–{chunk.char_end} · ~{chunk.token_estimate} tokens_",
                "",
                chunk.text,
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"
