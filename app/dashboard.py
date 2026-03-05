"""
Intervista Scoring Dashboard — visualize IQR results for stakeholders.

Run from project root: streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import html
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

# Project root (app/dashboard.py -> app -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "transcripts"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

# Dimension display order (match common IQR 5 dimensions)
DIMENSION_ORDER = [
    "Question Formulation",
    "Probing Quality",
    "Active Listening",
    "Rapport-Building",
    "Ethical Conduct",
]


def discover_transcripts() -> List[str]:
    """Recursively find all .json files under data/transcripts/; return relative paths (e.g. gold/1.json)."""
    if not TRANSCRIPTS_DIR.is_dir():
        return []
    paths = []
    for p in TRANSCRIPTS_DIR.rglob("*.json"):
        if p.is_file():
            rel = p.relative_to(TRANSCRIPTS_DIR)
            paths.append(str(rel).replace("\\", "/"))
    return sorted(paths)


def relative_path_to_evaluation_basename(relative_path: str) -> str:
    """Map e.g. gold/1.json -> gpt4o_gold_1_results.json (filename only)."""
    path = Path(relative_path)
    level = path.parent.name if path.parent.name else path.stem
    stem = path.stem
    return f"gpt4o_{level}_{stem}_results.json"


def load_json(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Score-based theming: (soft_border_color, deep_score_text_color, band_label)
SCORE_1_2 = ("#fee2e2", "#991b1b", "Needs Improvement")   # Soft Red border, Deep Red text
SCORE_3_4 = ("#fef3c7", "#92400e", "Proficient")         # Soft Amber border, Deep Amber text
SCORE_5_7 = ("#d1fae5", "#065f46", "Excellent")          # Soft Green border, Deep Green text


def score_theme(score: int) -> tuple[str, str, str]:
    if score <= 2:
        return SCORE_1_2
    if score <= 4:
        return SCORE_3_4
    return SCORE_5_7


# Icons for dimension cards (WPI-academic professional)
DIMENSION_ICONS = {
    "Question Formulation": "📋",
    "Probing Quality": "🔍",
    "Active Listening": "👂",
    "Rapport-Building": "🤝",
    "Ethical Conduct": "⚖️",
}


def render_metric_card(label: str, score: int) -> None:
    border_color, score_color, band_label = score_theme(score)
    icon = DIMENSION_ICONS.get(label, "•")
    label_esc = html.escape(label)
    band_esc = html.escape(band_label)
    st.markdown(
        f"""
        <div class="metric-card" style="
            background: #ffffff;
            border: 2px solid {border_color};
            border-radius: 10px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        ">
            <div style="font-size: 0.85rem; font-weight: 600; color: #1e293b; margin-bottom: 0.35rem;">{icon} {label_esc}</div>
            <div style="font-size: 2.25rem; font-weight: 800; color: {score_color}; line-height: 1.2;">{score}</div>
            <div style="font-size: 0.8rem; font-weight: 600; color: #1e293b;">{band_esc}</div>
            <div style="font-size: 0.7rem; color: #64748b;">/ 7</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_light_theme_css() -> None:
    """Inject WPI-academic professional light theme: off-white background, white blocks, deep charcoal text."""
    st.markdown(
        """
        <style>
        /* Overall theme: clean light background, sans-serif */
        .stApp { background-color: #f8fafc !important; }
        .stApp header { background: #ffffff !important; border-bottom: 1px solid #e2e8f0 !important; }
        main { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important; }
        main .stMarkdown { color: #1e293b !important; }
        main h1 { color: #0f172a !important; font-weight: 700 !important; }
        main h2, main h3 { color: #1e293b !important; font-weight: 600 !important; }
        main p, main li { color: #334155 !important; line-height: 1.6 !important; }
        /* Sidebar: mid-gray, deep blue title, charcoal labels */
        [data-testid="stSidebar"] { background: #f1f5f9 !important; border-right: 1px solid #e2e8f0 !important; }
        [data-testid="stSidebar"] h1 { color: #1e40af !important; font-weight: 700 !important; }
        [data-testid="stSidebar"] .stMarkdown { color: #334155 !important; }
        [data-testid="stSidebar"] label { color: #1e293b !important; font-weight: 500 !important; }
        /* Expanders: white content, dark text */
        .streamlit-expanderContent { background: #ffffff !important; color: #1e293b !important; border: 1px solid #e5e7eb !important; border-radius: 8px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Intervista IQR Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_light_theme_css()

    # Sidebar: polished structure — deep blue title, charcoal labels (Phase 4.2)
    st.sidebar.markdown(
        "<h1 style='color: #1e40af; font-weight: 700; margin-bottom: 0.5rem;'>Intervista Dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<p style='color: #1e293b; font-size: 0.85rem; font-weight: 600; margin-top: 0.5rem;'>Interview selector</p>",
        unsafe_allow_html=True,
    )
    transcript_options = discover_transcripts()
    if not transcript_options:
        st.sidebar.warning("No transcripts found under `data/transcripts/`.")
        st.error("**No interviews to display.** Add `.json` transcript files under `data/transcripts/` (e.g. `gold/1.json`, `bronze/2.json`).")
        return

    selected_relative = st.sidebar.selectbox(
        "Select an interview",
        options=transcript_options,
        index=0,
        help="List is built from data/transcripts/.",
        label_visibility="collapsed",
    )
    st.sidebar.markdown(
        "<p style='color: #1e293b; font-size: 0.8rem; margin-top: 1rem; font-weight: 600;'>Intel Trends</p>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Select an interview to view scoring and trends.")
    # Map selection to paths: relative path -> transcript path, evaluation path
    transcript_path = TRANSCRIPTS_DIR / selected_relative
    evaluation_basename = relative_path_to_evaluation_basename(selected_relative)
    evaluation_path = OUTPUT_DIR / evaluation_basename

    transcript_data = load_json(transcript_path)
    evaluation_data = load_json(evaluation_path)

    if transcript_data is None:
        st.error(f"**Missing transcript.** File not found: `{transcript_path}`")
        st.info("Ensure the file exists under `data/transcripts/`.")
        return

    if evaluation_data is None:
        st.warning("**Evaluation is still in progress or missing.**")
        st.info(
            f"No evaluation file found: `data/output/{evaluation_basename}`. "
            "Run the calibration script for this transcript, e.g.: "
            "`python tests/scripts/run_calibration.py --transcript data/transcripts/..."
            " --provider openai --model gpt-4o`"
        )
        # Show transcript-only view when evaluation is missing (light theme)
        meta = transcript_data.get("metadata") or {}
        st.markdown(
            f"""
            <div style="background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:1.25rem; margin-bottom:1rem; box-shadow:0 1px 3px rgba(0,0,0,0.06);">
                <h3 style="color:#1e293b;">Session metadata (transcript only)</h3>
                <p style="color:#334155;"><strong>Scenario:</strong> {html.escape(str(meta.get('scenario', '—')))}</p>
                <p style="color:#334155;"><strong>Persona:</strong> {html.escape(str(meta.get('persona', '—')))}</p>
                <p style="color:#334155;"><strong>Interview:</strong> <code>{html.escape(selected_relative)}</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("View raw transcript"):
            for t in transcript_data.get("turns") or []:
                st.markdown(f"**Turn {t.get('turn_id')} — {t.get('speaker', '?')}**")
                st.markdown(f"> {t.get('text', '')}")
        st.sidebar.markdown("---")
        st.sidebar.caption("Intervista IQR · Academic Dashboard")
        return

    meta = evaluation_data.get("metadata") or transcript_data.get("metadata") or {}
    scenario = meta.get("scenario", "—")
    persona = meta.get("persona", "—")
    overall_summary = evaluation_data.get("overall_summary", "")
    results: List[Dict[str, Any]] = evaluation_data.get("evaluation_results") or []

    # Header: white content blocks, deep charcoal / slate text
    st.title("Intervista IQR Scoring Results")
    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            """
            <div style="
                background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px;
                padding: 1.25rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            ">
                <h3 style="color: #1e293b; font-weight: 600; margin-top: 0;">Session metadata</h3>
                <p style="color: #334155; margin: 0.5rem 0;"><strong>Scenario:</strong> """ + html.escape(str(scenario)) + """</p>
                <p style="color: #334155; margin: 0.5rem 0;"><strong>Persona:</strong> """ + html.escape(str(persona)) + """</p>
                <p style="color: #334155; margin: 0.5rem 0;"><strong>Interview:</strong> <code>""" + html.escape(selected_relative) + """</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        summary_esc = html.escape(overall_summary).replace("\n", "<br/>")
        st.markdown(
            f"""
            <div style="
                background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px;
                padding: 1.25rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            ">
                <h3 style="color: #1e293b; font-weight: 600; margin-top: 0;">Overall summary</h3>
                <p style="color: #334155; line-height: 1.6;">{summary_esc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Dimension scores")
    # Build dimension -> result map; preserve order (Questioning, Probing, Listening, Rapport, Ethics)
    by_dim = {r.get("dimension_name"): r for r in results if r.get("dimension_name")}
    ordered = [by_dim[d] for d in DIMENSION_ORDER if d in by_dim]
    for r in results:
        if r.get("dimension_name") not in DIMENSION_ORDER:
            ordered.append(r)

    # Single horizontal row of 5 metric cards above Detailed Feedback
    cols = st.columns(5)
    for i, res in enumerate(ordered[:5]):
        with cols[i]:
            render_metric_card(res.get("dimension_name", "—"), res.get("score", 0))

    st.subheader("Detailed feedback")

    for res in ordered:
        dim_name = res.get("dimension_name", "Dimension")
        score = res.get("score", 0)
        label = res.get("label", "")
        rationale = res.get("rationale", "")
        evidence = res.get("evidence") or {}
        student_quote = evidence.get("student_quote", "")
        stakeholder_cue = evidence.get("stakeholder_cue", "")
        turn_id = evidence.get("turn_id", "")

        with st.expander(f"**{dim_name}** — Score: {score} ({label})"):
            st.markdown(
                "<div style='background:#fff; color:#1e293b; padding:0.5rem 0;'>"
                "<strong style='color:#1e293b;'>Rationale</strong></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='color:#334155; line-height: 1.6;'>{html.escape(rationale).replace(chr(10), '<br/>')}</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='background:#fff; color:#1e293b; padding:0.75rem 0 0.25rem 0;'>"
                "<strong style='color:#1e293b;'>Evidence comparison</strong></div>",
                unsafe_allow_html=True,
            )
            sq_esc = html.escape(student_quote).replace("\n", "<br/>")
            sc_esc = html.escape(stakeholder_cue).replace("\n", "<br/>")
            st.markdown(
                f"""
                <div style="display: grid; gap: 0.75rem; margin-top: 0.5rem;">
                    <div style="
                        padding: 1rem; background: #e0f2fe; border: 1px solid #bae6fd;
                        border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
                    ">
                        <div style="font-weight: 700; color: #0c4a6e; margin-bottom: 0.5rem;">Student (turn {turn_id})</div>
                        <div style="color: #1e293b; line-height: 1.5;">{sq_esc}</div>
                    </div>
                    <div style="
                        padding: 1rem; background: #fce7f3; border: 1px solid #fbcfe8;
                        border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
                    ">
                        <div style="font-weight: 700; color: #831843; margin-bottom: 0.5rem;">Stakeholder cue</div>
                        <div style="color: #1e293b; line-height: 1.5;">{sc_esc}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Transcript viewer")
    show_transcript = st.checkbox("Show full raw dialogue", value=False)
    if show_transcript:
        turns = transcript_data.get("turns") or []
        parts = [
            "<div style='background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:1.25rem; margin-top:0.5rem; box-shadow:0 1px 3px rgba(0,0,0,0.06);'>"
        ]
        for t in turns:
            tid = t.get("turn_id", "")
            speaker = html.escape(str(t.get("speaker", "?")))
            text = html.escape(str(t.get("text", ""))).replace("\n", "<br/>")
            parts.append(f"<p style='color:#1e293b; font-weight:600; margin:0.75rem 0 0.25rem 0;'>Turn {tid} — {speaker}</p>")
            parts.append(f"<p style='color:#334155; margin-left:1rem; line-height:1.5;'>{text}</p>")
        parts.append("</div>")
        st.markdown("".join(parts), unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.caption("Intervista IQR · Academic Dashboard")


if __name__ == "__main__":
    main()
