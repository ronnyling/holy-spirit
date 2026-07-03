"""Local Streamlit UI for the Knowledge Engine.

Three tabs:
  Ingest        — paste raw text as a transcript; runs the full pipeline
                  (harbour → extract → gap check → conflict → embed).
  Chat          — RAG chat grounded in the evidence-gated knowledge base.
                  Retrieves the most relevant claims, then asks MiMo to
                  answer within that evidence. Epistemic status is preserved.
  Knowledge     — live snapshot, domain browser, entity lookup.

Run from the knowledge_engine/ directory:
    pip install streamlit
    streamlit run app.py

All backend config is read from .env (same file the CLI uses).
Neo4j + Ollama bge-m3 must be running for full functionality; the UI
degrades gracefully and tells you exactly what is unavailable.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

from knowledge_engine.bootstrap import build_engine_from_env, load_dotenv
from knowledge_engine.contracts import TranscriptInput
from knowledge_engine.llm import MiMoClient



# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Knowledge Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── cached resources (built once per Streamlit server start) ──────────────────

@st.cache_resource(show_spinner="Connecting to Neo4j…")
def _get_engine():
    """Returns (engine, error_str).  engine is None when startup fails."""
    try:
        return build_engine_from_env(), None
    except SystemExit as exc:
        return None, f"SystemExit: {exc}"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_resource(show_spinner=False)
def _get_mimo() -> MiMoClient | None:
    load_dotenv()
    return MiMoClient.from_env()


engine, engine_error = _get_engine()
mimo = _get_mimo()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Knowledge Engine")
    st.caption("Evidence-gated R&D knowledge system")
    st.divider()

    if engine_error:
        st.error(f"Engine offline\n\n{engine_error}", icon="🔴")
        st.info(
            "Make sure Neo4j is running and `KE_NEO4J_*` env vars are set in `.env`.",
            icon="ℹ️",
        )
    else:
        st.success("Engine connected", icon="🟢")
        snap = engine.state_snapshot()
        # snapshot keys differ between graph-primary and JSON-store paths
        st.metric("Entities", snap.get("entities", snap.get("Entity", 0)))
        st.metric("Claims", snap.get("claims", snap.get("Claim", 0)))
        confirmed = snap.get("confirmed_claims", snap.get("Confirmed", "—"))
        st.metric("Confirmed", confirmed)

    st.divider()

    if mimo is None:
        st.warning("MiMo not configured\n\nSet `KE_MIMO_API_KEY` in `.env`.", icon="⚠️")
    else:
        st.success("MiMo ready", icon="💬")
        st.caption("Chat and domain classification available.")

# ── tabs ──────────────────────────────────────────────────────────────────────
# ── session state initialisation ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_ingest, tab_chat, tab_kb = st.tabs(["📥 Ingest", "💬 Chat", "📚 Knowledge Base"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — INGEST
# ─────────────────────────────────────────────────────────────────────────────
with tab_ingest:
    st.header("Add a Transcript")
    st.caption(
        "Paste a transcript below, or upload one or more `.txt` files. "
        "The system classifies the domain and entity automatically — "
        "just provide the raw content and run the pipeline."
    )

    _MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB per file

    raw_text = st.text_area(
        "Paste transcript text",
        height=320,
        placeholder="Paste the full transcript or notes here…",
    )
    uploaded_files = st.file_uploader(
        "Or upload .txt transcript files (max 5 MB each)",
        type=["txt"],
        accept_multiple_files=True,
        help="Select one or more plain-text files. Each file is processed as a separate transcript.",
    )

    can_ingest = engine is not None
    if st.button("▶ Run ingest pipeline", type="primary", disabled=not can_ingest):
        texts_to_process: list[tuple[str, str]] = []
        if raw_text.strip():
            texts_to_process.append(("pasted text", raw_text.strip()))
        for f in uploaded_files or []:
            if f.size > _MAX_FILE_BYTES:
                st.warning(f"⚠️ `{f.name}` exceeds the 5 MB limit — skipped.")
                continue
            try:
                file_text = f.read().decode("utf-8", errors="replace")
            except Exception as exc:
                st.warning(f"⚠️ Could not read `{f.name}`: {exc}")
                continue
            if file_text.strip():
                texts_to_process.append((f.name, file_text.strip()))

        if not texts_to_process:
            st.warning("Paste some text or upload at least one .txt file.")
        else:
            for label, text in texts_to_process:
                if len(texts_to_process) > 1:
                    st.markdown(f"---\n**Processing: {label}**")
                content_hash = str(abs(hash(text)))[:12]
                transcript = TranscriptInput(
                    transcript_text=text,
                    source_id=f"ui-{content_hash}",
                )
                with st.spinner(
                    f"Running pipeline for `{label}` "
                    "(classify → extract → gap check → conflict check → embed)…"
                    "\n\nThis may take 10–60 s depending on transcript length."
                ):
                    try:
                        outcome = engine.ingest_transcript(transcript)
                    except Exception as exc:
                        st.error(f"Ingest failed for `{label}`: {type(exc).__name__}: {exc}")
                        continue

                st.success(
                    f"Ingested → entity `{outcome.entity_id}`"
                    + (f" | transcript `{outcome.transcript_id}`" if outcome.transcript_id else "")
                )

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Claims extracted", len(outcome.claim_ids))
                m2.metric("Confirmed", len(outcome.confirmed_claim_ids))
                m3.metric("Unverified", len(outcome.unverified_claim_ids))
                m4.metric("Gaps flagged", len(outcome.gap_flags))
                m5.metric("Conflicts", len(outcome.conflict_summaries))

                if outcome.gap_flags:
                    with st.expander(f"⚠️ {len(outcome.gap_flags)} gap(s) — clarification needed"):
                        for g in outcome.gap_flags:
                            severity_icon = {"high": "🔴", "medium": "🟡", "low": "⚪"}.get(g.severity, "⚪")
                            st.markdown(
                                f"{severity_icon} **{g.kind}** — {g.question}"
                                + (f"\n\n> {g.rationale}" if g.rationale else "")
                            )

                if outcome.conflict_summaries:
                    with st.expander(f"⚡ {len(outcome.conflict_summaries)} conflict(s) detected"):
                        for c in outcome.conflict_summaries:
                            st.markdown(
                                f"- Signature: `{c.conflict_signature}` | "
                                f"Incoming: `{c.incoming_claim_id}`"
                            )

                if outcome.slot_suggestions:
                    with st.expander(f"💡 {len(outcome.slot_suggestions)} slot suggestion(s)"):
                        for s in outcome.slot_suggestions:
                            st.markdown(
                                f"- **{s.slot_name}**: `{s.current_lifecycle}` → "
                                f"`{s.suggested_lifecycle}` *(seen {s.observed_count}×)* — {s.reason}"
                            )

                if outcome.notes:
                    with st.expander("Pipeline notes"):
                        for n in outcome.notes:
                            st.write(f"- {n}")

                if outcome.claim_ids:
                    with st.expander(f"📋 View extracted claims ({len(outcome.claim_ids)})"):
                        try:
                            for cid in outcome.claim_ids[:30]:
                                detail = engine.get_claim(cid)
                                if detail and not detail.get("error"):
                                    claim_data = detail.get("claim", detail)
                                    status = claim_data.get("epistemic_status", "?")
                                    badge = {"Confirmed": "🟢", "Disputed": "🔴"}.get(status, "🟡")
                                    st.markdown(f"{badge} `{status}` — {claim_data.get('statement', cid)}")
                            if len(outcome.claim_ids) > 30:
                                st.caption(f"… and {len(outcome.claim_ids) - 30} more.")
                        except Exception:
                            st.caption("(claim detail unavailable — check Neo4j connection)")

                    st.info("💬 Ready to ask questions? Switch to the **Chat** tab.", icon="👉")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — CHAT
# ─────────────────────────────────────────────────────────────────────────────
with tab_chat:
    st.header("Research Chat")
    st.caption(
        "Ask anything. The engine retrieves the most relevant evidence-gated claims "
        "from the knowledge base and grounds MiMo's answer in them. "
        "Confirmed vs Unverified status is preserved — the LLM cannot invent knowledge "
        "that isn't in the system."
    )

    chat_ready = engine is not None and mimo is not None
    if not chat_ready:
        if engine is None:
            st.info("Chat requires a working engine connection (Neo4j offline).")
        else:
            st.info("Chat requires MiMo — set `KE_MIMO_API_KEY` in `.env`.")

    chat_mode = st.radio(
        "Mode",
        ["💬 Research Chat", "🧠 Explore Experience"],
        horizontal=True,
        help=(
            "**Research Chat**: retrieves matching claims from your knowledge base and "
            "grounds the answer in them.\n\n"
            "**Explore Experience**: combines what the world generally knows with your "
            "system's accumulated experience to produce a discerned, opinionated view."
        ),
    )

    with st.expander("⚙️ Chat settings", expanded=False):
        # Build domain list from ingested tags (what's actually in the graph)
        # rather than just policy names (which may have underscore/space drift).
        try:
            _chat_ingested_domains = engine.store.list_domains() if engine else []
        except Exception:
            _chat_ingested_domains = []
        from knowledge_engine.policy import list_policy_domains as _list_policy_domains
        _chat_domain_options = ["— all domains —"] + sorted(
            {d.lower().replace(" ", "_") for d in _chat_ingested_domains}
            | {d.lower().replace(" ", "_") for d in (_list_policy_domains() if engine else [])}
        )
        filter_domain = st.selectbox(
            "Filter domain (optional)",
            _chat_domain_options,
            key="chat_domain_filter",
        )
        domain_filter = (
            None if filter_domain == "— all domains —" else filter_domain
        )

    # Render conversation history.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])} claims)"):
                    for src in msg["sources"]:
                        badge = (
                            "🟢"
                            if src.get("epistemic_status") == "Confirmed"
                            else ("🔴" if src.get("epistemic_status") == "Disputed" else "🟡")
                        )
                        tags = src.get("tags", [])
                        tag_str = f" `{'`, `'.join(tags)}`" if tags else ""
                        sim = src.get("similarity")
                        sim_str = f" *(sim {sim:.2f})*" if sim is not None else ""
                        st.markdown(
                            f"{badge} **{src.get('epistemic_status', '?')}**"
                            f"{tag_str}{sim_str}  \n{src.get('statement', '')}"
                        )

    if chat_ready:
        if prompt := st.chat_input("Ask the knowledge base…"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):

                # ── Explore Experience mode ───────────────────────────────────
                if chat_mode == "🧠 Explore Experience":
                    with st.spinner("Fetching world knowledge and system experience…"):
                        try:
                            exp = engine.explore_experience(prompt, domain=domain_filter)
                        except Exception as exc:
                            exp = {"error": f"{type(exc).__name__}: {exc}"}

                    if exp.get("error"):
                        answer = f"⚠️ {exp['error']}"
                        exp_sources: list[dict] = []
                    else:
                        answer = exp.get("synthesis", "")
                        exp_sources = exp.get("experience_claims", [])

                        # Show the structured breakdown before the synthesis.
                        if not exp.get("experience_available"):
                            st.info(
                                "No system experience found yet — showing world knowledge only. "
                                "Ingest transcripts to build up experience.",
                                icon="ℹ️",
                            )
                        else:
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Confirmed", exp.get("confirmed_count", 0))
                            c2.metric("Unverified", exp.get("unverified_count", 0))
                            c3.metric("Disputed", exp.get("disputed_count", 0))

                        with st.expander("🌍 World knowledge baseline", expanded=False):
                            st.markdown(exp.get("world_knowledge", ""))

                        st.markdown("### 🧠 Discerned Position")
                        st.markdown(answer)

                        if exp_sources:
                            with st.expander(
                                f"📎 Experience ({len(exp_sources)} claims)",
                                expanded=True,
                            ):
                                for src in exp_sources:
                                    status = src.get("epistemic_status") or src.get("status") or "Unknown"
                                    badge = {"Confirmed": "🟢", "Disputed": "🔴"}.get(status, "🟡")
                                    tags = src.get("tags") or []
                                    tag_str = f" `{'`, `'.join(tags)}`" if tags else ""
                                    sim = src.get("similarity") or src.get("score")
                                    sim_str = f" *(sim {sim:.3f})*" if sim is not None else ""
                                    st.markdown(
                                        f"{badge} **{status}**{tag_str}{sim_str}  \n{src.get('statement', '')}"
                                    )

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": exp_sources}
                    )

                # ── Research Chat (standard RAG) mode ────────────────────────
                else:
                    sources: list[dict] = []
                    context_lines: list[str] = []

                    search_error: str | None = None
                    with st.spinner("Searching knowledge base…"):
                        try:
                            result = engine.search_claims(
                                prompt, domain=domain_filter, limit=6
                            )
                            if result.get("error"):
                                search_error = result["error"]
                            else:
                                raw_hits = result.get("claims", result.get("matches", []))
                                for h in raw_hits:
                                    stmt = h.get("statement", "")
                                    status = h.get("epistemic_status") or h.get("status") or "Unknown"
                                    tags = h.get("tags") or []
                                    sim = h.get("similarity") or h.get("score")
                                    sources.append(
                                        {
                                            "statement": stmt,
                                            "epistemic_status": status,
                                            "tags": tags,
                                            "similarity": round(float(sim), 3) if sim is not None else None,
                                        }
                                    )
                                    tag_str = ", ".join(tags) if tags else "unknown domain"
                                    context_lines.append(f"[{status}] {stmt}  (domain: {tag_str})")
                        except Exception as exc:
                            search_error = f"{type(exc).__name__}: {exc}"

                    context_block = "\n".join(context_lines) if context_lines else "(no relevant claims found)"

                    if search_error:
                        st.warning(f"Knowledge base search failed: {search_error}", icon="⚠️")
                    elif sources:
                        st.caption(
                            f"🔍 Retrieved **{len(sources)}** claim(s) from the knowledge base"
                            + (f" (filtered to `{domain_filter}`)" if domain_filter else "")
                            + " — grounding answer below."
                        )
                    else:
                        st.caption(
                            f"🔍 **0 claims retrieved** from the knowledge base"
                            + (f" for domain `{domain_filter}`" if domain_filter else "")
                            + " — answer will reflect that gap."
                        )

                    system_prompt = (
                        "You are a research assistant with access to a curated, "
                        "evidence-gated knowledge base built from expert transcripts.\n\n"
                        "Rules:\n"
                        "1. Ground your answer in the knowledge base context provided below.\n"
                        "2. For each point you make, indicate whether it is Confirmed "
                        "   (evidence-backed) or Unverified (observed but not yet proven).\n"
                        "3. If the knowledge base has no relevant information on a point, "
                        "   say so explicitly — do not fabricate or extrapolate.\n"
                        "4. Preserve dissent: if the knowledge base contains conflicting "
                        "   positions, present both and note the disagreement.\n"
                        "5. For TCM and financial topics, always attribute claims to their "
                        "   source — never present synthesised knowledge as anonymous fact.\n"
                        "6. Unknown is valid — if a question cannot be answered from the "
                        "   knowledge base, say the answer is currently unknown."
                    )

                    user_prompt = (
                        f"Question: {prompt}\n\n"
                        f"Knowledge base context ({len(sources)} claim(s) retrieved):\n"
                        f"{context_block}"
                    )

                    with st.spinner("Asking MiMo…"):
                        try:
                            answer = mimo.complete_sync(
                                system=system_prompt,
                                user=user_prompt,
                                max_tokens=2000,
                            )
                        except Exception as exc:
                            answer = f"⚠️ LLM call failed: {type(exc).__name__}: {exc}"
                            sources = []

                    st.markdown(answer)

                    if sources:
                        with st.expander(
                            f"📎 Sources ({len(sources)} claims from knowledge base)",
                            expanded=True,
                        ):
                            for src in sources:
                                badge = (
                                    "🟢"
                                    if src.get("epistemic_status") == "Confirmed"
                                    else ("🔴" if src.get("epistemic_status") == "Disputed" else "🟡")
                                )
                                tags = src.get("tags") or []
                                tag_str = f" `{'`, `'.join(tags)}`" if tags else ""
                                sim = src.get("similarity")
                                sim_str = f" *(sim {sim:.3f})*" if sim is not None else ""
                                st.markdown(
                                    f"{badge} **{src['epistemic_status']}**"
                                    f"{tag_str}{sim_str}  \n{src['statement']}"
                                )

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )

    if st.session_state.get("messages"):
        st.divider()
        if st.button("🗑 Clear chat history"):
            st.session_state.messages = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────
with tab_kb:
    st.header("Knowledge Base")

    if engine is None:
        st.info("Engine offline — no data to display.")
    else:
        snap = engine.state_snapshot()
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Entities", snap.get("entities", snap.get("Entity", 0)))
        m2.metric("Claims", snap.get("claims", snap.get("Claim", 0)))
        m3.metric("Confirmed", snap.get("confirmed_claims", snap.get("Confirmed", 0)))
        m4.metric("Evidence", snap.get("evidence", snap.get("Evidence", 0)))
        m5.metric("Open cases", snap.get("open_cases", "—"))

        st.divider()

        # ── pending slot promotions ───────────────────────────────────────────
        try:
            pending = engine.list_pending_promotions()
        except Exception:
            pending = []

        promo_label = (
            f"💡 Pending Slot Promotions ({len(pending)})"
            if pending
            else "💡 Pending Slot Promotions"
        )
        with st.expander(promo_label, expanded=bool(pending)):
            if not pending:
                st.success("No pending promotions — slot queue is empty.", icon="✅")
            else:
                st.caption(
                    "The slot learner detected these candidates while ingesting transcripts. "
                    "Review and confirm to advance the slot lifecycle, or leave them pending."
                )
                for item in pending:
                    entity_id = item.get("entity_id", "?")
                    slot_name = item.get("slot_name", "?")
                    cur = item.get("current_lifecycle", "?")
                    sug = item.get("suggested_lifecycle", "?")
                    cnt = item.get("observed_count", 0)
                    reason = item.get("reason", "")
                    with st.container(border=True):
                        st.markdown(
                            f"**{slot_name}** &nbsp; `{cur}` → `{sug}` &nbsp; "
                            f"*(seen {cnt}×)*  \n{reason}"
                        )
                        st.caption(f"Entity: `{entity_id}`")
                        col_who, col_btn = st.columns([3, 1])
                        confirmed_by = col_who.text_input(
                            "Your name",
                            placeholder="Enter your name to confirm",
                            key=f"promo_who_{entity_id}_{slot_name}",
                            label_visibility="collapsed",
                        )
                        if col_btn.button(
                            "✅ Confirm",
                            key=f"promo_btn_{entity_id}_{slot_name}",
                            disabled=not confirmed_by.strip(),
                        ):
                            try:
                                from knowledge_engine.models import SlotLifecycle
                                target = SlotLifecycle(sug)
                                entity_name = item.get("entity_name") or entity_id
                                engine.confirm_slot(
                                    entity_name=entity_name,
                                    slot_name=slot_name,
                                    confirmed_by=confirmed_by.strip(),
                                    target=target,
                                )
                                st.success(f"Promoted `{slot_name}` → `{sug}`")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Promotion failed: {exc}")

        st.divider()

        # ── open conflicts ────────────────────────────────────────────────────
        open_count = snap.get("open_cases", 0)
        section_label = (
            f"⚡ Open Conflicts ({open_count})"
            if open_count
            else "⚡ Open Conflicts"
        )
        with st.expander(section_label, expanded=bool(open_count)):
            if not open_count:
                st.success("No open conflicts — all cases resolved.", icon="✅")
            else:
                st.caption(
                    "Each conflict is a case where an incoming claim contradicts existing "
                    "knowledge. Enter your decision and rationale; the resolution is stored "
                    "as memory and reused automatically for similar future conflicts."
                )
                open_cases = engine.list_open_cases()
                if not open_cases:
                    st.info("Cases detected in snapshot but not retrievable — check Neo4j.")
                for case in open_cases:
                    case_id = case["case_id"]
                    sig = case["conflict_signature"]
                    claims = case["claims"]
                    notes = case.get("research_notes") or ""

                    with st.container(border=True):
                        st.markdown(f"**Case** `{case_id[:16]}…`  \nSignature: `{sig}`")

                        if notes:
                            st.caption(f"Research notes: {notes}")

                        # Show conflicting claims side by side (up to 4 columns).
                        if claims:
                            cols = st.columns(min(len(claims), 4))
                            for i, cl in enumerate(claims[:4]):
                                badge = {
                                    "Confirmed": "🟢", "Disputed": "🔴", "Retracted": "⬛"
                                }.get(cl["epistemic_status"], "🟡")
                                cols[i].markdown(
                                    f"{badge} `{cl['epistemic_status']}`  \n{cl['statement']}"
                                )
                            if len(claims) > 4:
                                st.caption(f"…and {len(claims) - 4} more conflicting claim(s).")

                        st.markdown("**Your resolution**")
                        col_d, col_r = st.columns([1, 2])
                        decision = col_d.selectbox(
                            "Decision",
                            [
                                "Accept incoming — replace existing",
                                "Reject incoming — keep existing",
                                "Both valid — context-dependent",
                                "Defer — needs more evidence",
                            ],
                            key=f"decision_{case_id}",
                        )
                        rationale = col_r.text_input(
                            "Rationale",
                            placeholder="Why? e.g. 'Newer data supersedes 2019 study'",
                            key=f"rationale_{case_id}",
                        )
                        if st.button(
                            "✅ Submit resolution",
                            key=f"resolve_{case_id}",
                            disabled=not rationale.strip(),
                        ):
                            try:
                                engine.resolve_case(case_id, decision=decision, rationale=rationale.strip())
                                st.success(
                                    f"Case `{case_id[:16]}…` resolved. "
                                    "The decision is stored as memory for future similar conflicts."
                                )
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Resolution failed: {exc}")

        st.divider()

        # ── domain browser ────────────────────────────────────────────────────
        st.subheader("Browse by domain")

        try:
            ingested_domains = engine.store.list_domains()
        except Exception:
            ingested_domains = []

        from knowledge_engine.policy import list_policy_domains as _list_kbpd
        all_domains = sorted(
            {d.lower().replace(" ", "_") for d in ingested_domains}
            | {d.lower().replace(" ", "_") for d in _list_kbpd()}
        )

        if all_domains:
            col_d, col_btn = st.columns([3, 1])
            chosen = col_d.selectbox("Domain", all_domains, key="kb_domain")
            load_clicked = col_btn.button("Load", key="kb_load")

            if load_clicked:
                with st.spinner(f"Loading {chosen}…"):
                    try:
                        result = engine.search_by_domain(chosen, limit=100)
                    except Exception as exc:
                        result = {"error": str(exc)}

                if result.get("error"):
                    st.error(result["error"])
                else:
                    claims = result.get("claims", [])
                    st.write(f"**{len(claims)} claim(s)** in domain `{chosen}`")
                    for cl in claims:
                        status = cl.get("epistemic_status", cl.get("status", "Unknown"))
                        badge = {"Confirmed": "🟢", "Disputed": "🔴", "Retracted": "⬛"}.get(
                            status, "🟡"
                        )
                        st.markdown(f"{badge} `{status}` — {cl.get('statement', '')}")
        else:
            st.info("No domains found yet — ingest some transcripts first.")

        st.divider()

        # ── entity lookup ─────────────────────────────────────────────────────
        st.subheader("Entity lookup")
        col_e, col_eb = st.columns([3, 1])
        entity_q = col_e.text_input(
            "Entity name",
            key="kb_entity",
            placeholder="e.g. Cap Rate Rules",
            label_visibility="collapsed",
        )
        lookup_clicked = col_eb.button("Look up", key="kb_lookup")

        if lookup_clicked and entity_q.strip():
            with st.spinner("Looking up…"):
                result = engine.get_entity(entity_name=entity_q.strip())
            if result.get("error"):
                st.warning(result["error"])
            else:
                # Show a clean summary before raw JSON.
                entity_data = result.get("entity", {})
                claims_data = result.get("claims", [])
                slots_data = result.get("slots", [])

                st.markdown(f"### {entity_data.get('canonical_name', entity_q)}")
                if entity_data.get("description"):
                    st.caption(entity_data["description"])
                if entity_data.get("aliases"):
                    st.caption(f"Also known as: {', '.join(entity_data['aliases'])}")

                c1, c2, c3 = st.columns(3)
                c1.metric("Claims", len(claims_data))
                c2.metric(
                    "Confirmed",
                    sum(1 for c in claims_data if c.get("status") == "Confirmed"),
                )
                c3.metric("Slots observed", len(slots_data))

                if claims_data:
                    with st.expander("Claims"):
                        for cl in claims_data:
                            status = cl.get("status", "Unknown")
                            badge = {"Confirmed": "🟢", "Disputed": "🔴", "Retracted": "⬛"}.get(
                                status, "🟡"
                            )
                            st.markdown(f"{badge} `{status}` — {cl.get('statement', '')}")

                if slots_data:
                    with st.expander("Slots"):
                        for s in slots_data:
                            st.markdown(
                                f"- **{s.get('name', '?')}** "
                                f"`{s.get('lifecycle', '?')}` "
                                f"(seen {s.get('observed_count', 0)}×)"
                            )

        st.divider()

        # ── cross-domain patterns ─────────────────────────────────────────────
        st.subheader("Cross-domain patterns")
        st.caption(
            "Surface claim pairs from different domains with high semantic similarity — "
            "a sign that distinct fields may share underlying principles."
        )

        col_sim, col_lim, col_pb = st.columns([2, 2, 1])
        min_sim = col_sim.slider("Min similarity", 0.5, 0.95, 0.7, step=0.05)
        xd_limit = col_lim.number_input("Max patterns", 5, 50, 15)
        xd_clicked = col_pb.button("Find patterns", key="kb_xd")

        if xd_clicked:
            with st.spinner("Scanning for cross-domain patterns…"):
                try:
                    patterns = engine.store.find_cross_domain_patterns(
                        min_similarity=min_sim, limit=int(xd_limit)
                    )
                except Exception as exc:
                    patterns = None
                    st.error(f"Pattern search failed: {exc}")

            if patterns is not None:
                if not patterns:
                    st.info("No cross-domain patterns found above the similarity threshold.")
                else:
                    st.write(f"**{len(patterns)} pattern(s) found**")
                    for p in patterns:
                        sim_val = round(float(p.get("similarity", 0)), 3)
                        ca = p.get("claim_a", {})
                        cb = p.get("claim_b", {})
                        with st.expander(f"sim {sim_val} — {ca.get('statement', '')[:60]}…"):
                            col_x, col_y = st.columns(2)
                            with col_x:
                                st.markdown(
                                    f"**A** `{ca.get('epistemic_status', '?')}`  \n"
                                    f"{ca.get('statement', '')}"
                                )
                            with col_y:
                                st.markdown(
                                    f"**B** `{cb.get('epistemic_status', '?')}`  \n"
                                    f"{cb.get('statement', '')}"
                                )
