"""
Intervista Scoring Dashboard — Diagnostic Coach UI for IQR results.

Run from project root: streamlit run app/dashboard.py
"""
from __future__ import annotations

import html
import json
from collections import Counter
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

# Default tactical actions when JSON omits next_interview_actions
DEFAULT_NEXT_ACTIONS = [
    "Ask at least 2 open-ended questions per topic.",
    "Follow up once on every stakeholder concern.",
    "Avoid interrupting stakeholder responses.",
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


def pretty_transcript_label(relative_path: str) -> str:
    """
    Human-friendly label for sidebar selector.
    Example: 'bronze/1.json' -> 'Bronze · Interview 1'.
    """
    path = Path(relative_path)
    level = path.parent.name or path.stem
    stem = path.stem
    return f"{level.capitalize()} · Interview {stem}"


def load_json(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# 10-point diagnostic band titles (aligned with IQR Phase 1 / schema skill_level_title)
def skill_title_from_mean(score: float) -> str:
    """Map a numeric mean to the canonical skill band title for session-level display."""
    if score >= 10.0:
        return "Master Stakeholder Partner"
    if score >= 9.0:
        return "Advanced Systems Interviewer"
    if score >= 8.0:
        return "Competent Operational Interviewer"
    if score >= 7.0:
        return "Emerging Technical Interviewer"
    if score >= 6.0:
        return "Novice Fact-Finder"
    return "Developing interviewer"


def _use_legacy_7point_scale(meta: Dict[str, Any], results: List[Dict[str, Any]]) -> bool:
    """True when JSON looks like pre–10-point IQR (integer 1–7, no diagnostic fields)."""
    if meta.get("iqr_score_scale") == "10-point" or meta.get("iqr_score_max") == 10.0:
        return False
    if any((r.get("skill_level_title") or "").strip() for r in results):
        return False
    return True


def parse_dimension_score(
    raw: Any,
    meta: Dict[str, Any] | None = None,
    results: List[Dict[str, Any]] | None = None,
) -> float:
    """Coerce JSON score to float on the 10-point scale for display."""
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.0
    meta = meta or {}
    results = results or []
    if _use_legacy_7point_scale(meta, results) and v == int(v) and 1 <= v <= 7:
        return round((v / 7.0) * 10.0, 1)
    return v


def session_skill_badge_parts(
    results: List[Dict[str, Any]], meta: Dict[str, Any] | None = None
) -> tuple[float, str]:
    """
    Build session-level headline: mean score and a representative skill_level_title
    (mode across dimensions, else derived from mean).
    """
    if not results:
        return 0.0, "—"
    scores = [parse_dimension_score(r.get("score"), meta, results) for r in results]
    mean_score = sum(scores) / len(scores)
    titles = [str(r.get("skill_level_title") or "").strip() for r in results]
    titles = [t for t in titles if t]
    if titles:
        mode_title = Counter(titles).most_common(1)[0][0]
        return round(mean_score, 1), mode_title
    return round(mean_score, 1), skill_title_from_mean(mean_score)


def coach_accent_colors(mean_score: float) -> Dict[str, str]:
    """Amber-forward for developing / novice; green-forward for advanced."""
    if mean_score >= 8.0:
        return {
            "header_bg": "linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 50%, #ffffff 100%)",
            "header_border": "#059669",
            "header_accent": "#047857",
            "badge_soft": "#d1fae5",
        }
    return {
        "header_bg": "linear-gradient(135deg, #fffbeb 0%, #fff7ed 50%, #ffffff 100%)",
        "header_border": "#d97706",
        "header_accent": "#b45309",
        "badge_soft": "#fef3c7",
    }


def missed_insight_text(res: Dict[str, Any]) -> str:
    """
    Plain text only (no HTML): what was lost — prefer line_of_inquiry_impact, else rationale.
    Escape before embedding in templates via escape_html_multiline().
    """
    li = res.get("line_of_inquiry_impact")
    if li is not None and str(li).strip():
        return str(li).strip()
    rat = (res.get("rationale") or "").strip()
    return rat if rat else "—"


def escape_html_multiline(plain: str) -> str:
    """Escape user/content for safe HTML; preserve line breaks as <br/>."""
    return html.escape(plain).replace("\n", "<br/>")


def dimension_insight_card_html(
    stakeholder_content_html: str,
    missed_insight_content_html: str,
    show_missed_insight: bool,
) -> str:
    """
    Full HTML for stakeholder cue and optional Missed insight.
    Pass only pre-escaped fragments (from escape_html_multiline); this function
    does not wrap content in code fences or <pre>.
    """
    missed_html = ""
    if show_missed_insight:
        missed_html = (
            '<div style="font-weight:700;color:#b45309;font-size:0.78rem;margin-bottom:0.35rem;">Missed insight</div>'
            f'<div style="color:#292524;line-height:1.55;font-size:0.95rem;">{missed_insight_content_html}</div>'
        )

    return (
        '<div style="background:#fafaf9;border:1px solid #e7e5e4;border-top:1px dashed #d6d3d1;'
        'border-radius:0 0 12px 12px;padding:1rem 1.15rem 1.15rem 1.15rem;margin-bottom:0.5rem;">'
        '<div style="font-weight:700;color:#57534e;font-size:0.78rem;margin-bottom:0.4rem;">Stakeholder cue</div>'
        f'<div style="color:#44403c;line-height:1.55;font-size:0.92rem;margin-bottom:1rem;">{stakeholder_content_html}</div>'
        f"{missed_html}"
        "</div>"
    )


def parse_next_interview_actions(evaluation_data: Dict[str, Any]) -> List[str]:
    """Normalize next_interview_actions from JSON (list, newline string, or defaults)."""
    raw = evaluation_data.get("next_interview_actions")
    if isinstance(raw, list):
        out = [str(x).strip() for x in raw if str(x).strip()]
        return out if out else list(DEFAULT_NEXT_ACTIONS)
    if isinstance(raw, str) and raw.strip():
        lines = [ln.strip().lstrip("•").lstrip("-").strip() for ln in raw.splitlines()]
        lines = [ln for ln in lines if ln]
        return lines if lines else list(DEFAULT_NEXT_ACTIONS)
    return list(DEFAULT_NEXT_ACTIONS)


def tactical_game_plan_three(evaluation_data: Dict[str, Any]) -> List[str]:
    """Exactly three bullets: JSON order first, then pad from DEFAULT_NEXT_ACTIONS."""
    base = parse_next_interview_actions(evaluation_data)
    out: List[str] = []
    seen: set[str] = set()
    for item in base:
        if len(out) >= 3:
            break
        if item not in seen:
            seen.add(item)
            out.append(item)
    for d in DEFAULT_NEXT_ACTIONS:
        if len(out) >= 3:
            break
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out[:3]


# Icons for dimension sections
DIMENSION_ICONS = {
    "Question Formulation": "📋",
    "Probing Quality": "🔍",
    "Active Listening": "👂",
    "Rapport-Building": "🤝",
    "Ethical Conduct": "⚖️",
}


# One-sentence description for each skill band shown below the score badge
SKILL_LEVEL_DESCRIPTIONS: Dict[str, str] = {
    "Novice Fact-Finder": (
        "A novice fact-finder can collect surface-level information but relies heavily on closed questions, "
        "rarely probes beyond the obvious, and tends to capture what happened without uncovering why."
    ),
    "Developing interviewer": (
        "Beginning to form structured questions but still inconsistent in probing and follow-through; "
        "often misses cues that warrant deeper exploration."
    ),
    "Emerging Technical Interviewer": (
        "Demonstrates growing command of open-ended questions and targeted probes, "
        "though depth and consistency still vary across topics."
    ),
    "Competent Operational Interviewer": (
        "Reliably structures sessions, uses probing effectively, and surfaces most relevant details; "
        "minor gaps remain in rapport and adaptive follow-up."
    ),
    "Advanced Systems Interviewer": (
        "Combines strong technique with contextual awareness, consistently drawing out root causes "
        "and stakeholder perspectives with minimal wasted turns."
    ),
    "Master Stakeholder Partner": (
        "Fluently adapts to any interview context, builds genuine rapport, and extracts nuanced insight "
        "that less experienced interviewers routinely miss."
    ),
}

# Glossary of common interview technique terms referenced in coach feedback
GLOSSARY: List[Dict[str, str]] = [
    {
        "term": "Leading question",
        "definition": (
            "A question that subtly steers the respondent toward a particular answer by embedding an assumption "
            "or preferred outcome (e.g., 'You were satisfied with the process, weren't you?'). "
            "Leading questions bias the data and should be replaced with neutral, open-ended alternatives."
        ),
    },
    {
        "term": "Double-barreled question",
        "definition": (
            "A single question that asks about two separate issues at once "
            "(e.g., 'Was the process clear and did you feel supported?'). "
            "The respondent cannot answer both parts accurately in one reply; split them into two distinct questions."
        ),
    },
    {
        "term": "Closed question",
        "definition": (
            "A question that invites only a yes/no or single-word answer, limiting the information gathered "
            "(e.g., 'Did you attend the meeting?'). Use closed questions sparingly—mainly to confirm facts."
        ),
    },
    {
        "term": "Open-ended question",
        "definition": (
            "A question that invites the respondent to elaborate freely "
            "(e.g., 'Can you walk me through what happened?'). "
            "Open-ended questions are the primary tool for uncovering context, reasoning, and detail."
        ),
    },
    {
        "term": "Probing question",
        "definition": (
            "A follow-up question that digs deeper into a previous response to surface root causes, "
            "nuance, or unstated assumptions (e.g., 'What led you to that conclusion?'). "
            "Effective probing distinguishes strong interviewers from those who merely collect facts."
        ),
    },
    {
        "term": "Rapport-building",
        "definition": (
            "Techniques used to establish trust and psychological safety with the interviewee—such as "
            "acknowledging responses, using the person's name, or briefly normalising difficult topics—"
            "so that they feel comfortable sharing openly."
        ),
    },
]


def inject_coach_theme_css() -> None:
    """Interview Diagnostic Coach theme: coach-oriented palette, quote styling, expanders."""
    st.markdown(
        """
        <style>
        .stApp { background-color: #fafaf9 !important; }
        .stApp header { background: #ffffff !important; border-bottom: 1px solid #e7e5e4 !important; }
        main { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important; }
        main .stMarkdown { color: #1c1917 !important; }
        main h1 { color: #0c0a09 !important; font-weight: 700 !important; }
        main h2, main h3 { color: #292524 !important; font-weight: 600 !important; }
        [data-testid="stSidebar"] {
          background: #f5f5f4 !important;
          border-right: 1px solid #e7e5e4 !important;
        }
        [data-testid="stSidebar"] h1 { color: #b45309 !important; font-weight: 700 !important; }
        [data-testid="stSidebar"] [data-testid="stSelectbox"] > div {
          border-radius: 999px !important;
          border: 1px solid #d6d3d1 !important;
          background: #1c1917 !important;
          box-shadow: 0 4px 12px rgba(28,25,23,0.25) !important;
        }
        [data-testid="stSidebar"] [data-testid="stSelectbox"] input {
          caret-color: transparent !important;
          user-select: none !important;
        }
        [data-testid="stSidebar"] [data-testid="stSelectbox"] div[role="button"],
        [data-testid="stSidebar"] [data-testid="stSelectbox"] span {
          color: #fafaf9 !important;
          font-weight: 600 !important;
        }
        [data-testid="stExpander"] {
          background: #ffffff !important;
          border: 1px solid #e7e5e4 !important;
          border-radius: 8px !important;
          margin-bottom: 0.5rem !important;
        }
        .streamlit-expanderHeader,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] label {
          background: #ffffff !important;
          color: #292524 !important;
          font-weight: 600 !important;
        }
        .streamlit-expanderContent {
          background: #ffffff !important;
          color: #292524 !important;
          border: 1px solid #e7e5e4 !important;
          border-radius: 8px !important;
        }
        /* Avoid Streamlit Markdown rendering indented HTML as <pre> / code blocks */
        div[data-testid="stMarkdownContainer"] pre {
          background: transparent !important;
          border: none !important;
          box-shadow: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Interview Diagnostic Coach — IQR",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_coach_theme_css()

    st.sidebar.markdown(
        "<h1 style='color: #b45309; font-weight: 700; margin-bottom: 0.25rem;'>Interview Diagnostic Coach</h1>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<p style='color: #44403c; font-size: 0.85rem; font-weight: 600;'>Interview selector</p>",
        unsafe_allow_html=True,
    )
    transcript_options = discover_transcripts()
    if not transcript_options:
        st.sidebar.warning("No transcripts found under `data/transcripts/`.")
        st.error(
            "**No interviews to display.** Add `.json` transcript files under `data/transcripts/` "
            "(e.g. `gold/1.json`, `bronze/2.json`)."
        )
        return

    selected_relative = st.sidebar.selectbox(
        "Select an interview",
        options=transcript_options,
        index=0,
        help="List is built from data/transcripts/.",
        label_visibility="collapsed",
        format_func=pretty_transcript_label,
    )
    st.sidebar.caption("Select a session to view coaching feedback.")

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
        meta = transcript_data.get("metadata") or {}
        st.markdown(
            f"""
            <div style="background:#fff; border:1px solid #e7e5e4; border-radius:10px; padding:1.25rem; margin-bottom:1rem;">
                <h3 style="color:#292524;">Session metadata (transcript only)</h3>
                <p style="color:#57534e;"><strong>Scenario:</strong> {html.escape(str(meta.get('scenario', '—')))}</p>
                <p style="color:#57534e;"><strong>Persona:</strong> {html.escape(str(meta.get('persona', '—')))}</p>
                <p style="color:#57534e;"><strong>Interview:</strong> <code>{html.escape(selected_relative)}</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("View raw transcript"):
            for t in transcript_data.get("turns") or []:
                st.markdown(f"**Turn {t.get('turn_id')} — {t.get('speaker', '?')}**")
                st.markdown(f"> {t.get('text', '')}")
        st.sidebar.markdown("---")
        st.sidebar.caption("Diagnostic Coach · IQR")
        return

    meta = evaluation_data.get("metadata") or transcript_data.get("metadata") or {}
    scenario = meta.get("scenario", "—")
    persona = meta.get("persona", "—")
    overall_summary = evaluation_data.get("overall_summary", "")
    results: List[Dict[str, Any]] = evaluation_data.get("evaluation_results") or []

    st.title("Interview Diagnostic Coach")
    st.markdown(
        "<p style='color:#57534e; font-size:1.05rem; margin-top:0;'>Interview Quality Rubric — feedback for your next session</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='margin-bottom: 0.75rem;'></div>", unsafe_allow_html=True)

    if results:
        mean_s, badge_title = session_skill_badge_parts(results, meta)
        colors = coach_accent_colors(mean_s)
        title_esc = html.escape(badge_title)
        desc = SKILL_LEVEL_DESCRIPTIONS.get(badge_title, "")
        desc_html = (
            f'<p style="color:#57534e; font-size:0.93rem; line-height:1.6; margin:0.6rem 0 0 0;">'
            f'{html.escape(desc)}</p>'
            if desc
            else ""
        )
        st.markdown(
            f"""
            <div style="
                background: {colors['header_bg']};
                border: 1px solid #e7e5e4;
                border-left: 6px solid {colors['header_border']};
                border-radius: 14px;
                padding: 1.5rem 1.75rem;
                margin-bottom: 1.35rem;
                box-shadow: 0 8px 24px rgba(28, 25, 23, 0.08);
            ">
                <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; color: #78716c;
                    text-transform: uppercase; margin-bottom: 0.4rem;">Skill level</div>
                <div style="font-size: 2rem; font-weight: 800; color: #0c0a09; line-height: 1.2; letter-spacing: -0.02em;">
                    <span style="color: {colors['header_accent']};">{mean_s:.1f}/10</span>
                    <span style="color: #a8a29e; font-weight: 600;"> — </span>
                    <span>{title_esc}</span>
                </div>
                {desc_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            f"""
            <div style="
                background: #ffffff; border: 1px solid #e7e5e4; border-radius: 12px;
                padding: 1.25rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            ">
                <h3 style="color: #292524; font-weight: 600; margin-top: 0;">Session</h3>
                <p style="color: #57534e; margin: 0.5rem 0;"><strong>Scenario:</strong> {html.escape(str(scenario))}</p>
                <p style="color: #57534e; margin: 0.5rem 0;"><strong>Persona:</strong> {html.escape(str(persona))}</p>
                <p style="color: #57534e; margin: 0.5rem 0;"><strong>Interview:</strong> <code>{html.escape(selected_relative)}</code></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        summary_esc = html.escape(overall_summary).replace("\n", "<br/>")
        st.markdown(
            f"""
            <div style="
                background: #ffffff; border: 1px solid #e7e5e4; border-radius: 12px;
                padding: 1.25rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            ">
                <h3 style="color: #292524; font-weight: 600; margin-top: 0;">Overall summary</h3>
                <p style="color: #57534e; line-height: 1.65;">{summary_esc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Dimension diagnostics")
    by_dim = {r.get("dimension_name"): r for r in results if r.get("dimension_name")}
    ordered = [by_dim[d] for d in DIMENSION_ORDER if d in by_dim]
    for r in results:
        if r.get("dimension_name") not in DIMENSION_ORDER:
            ordered.append(r)

    for res in ordered:
        dim_name = res.get("dimension_name", "Dimension")
        score_raw = res.get("score", 0)
        score = parse_dimension_score(score_raw, meta, results)
        label = res.get("label", "")
        skill_dim = (res.get("skill_level_title") or "").strip()
        rationale = res.get("rationale", "")
        evidence = res.get("evidence") or {}
        student_quote = evidence.get("student_quote", "")
        alt_phrasing = evidence.get("alternative_phrasing")
        stakeholder_cue = evidence.get("stakeholder_cue", "")
        turn_id = evidence.get("turn_id", "")
        dim_colors = coach_accent_colors(score)

        icon = DIMENSION_ICONS.get(dim_name, "•")
        dim_esc = html.escape(dim_name)
        label_esc = html.escape(str(label))
        skill_esc = html.escape(skill_dim) if skill_dim else ""
        skill_suffix = (
            f'<span style="font-size:0.88rem; font-weight:600; color:#78716c;"> · {skill_esc}</span>'
            if skill_esc
            else ""
        )

        st.markdown(
            f"""
            <div style="
                background: #ffffff; border: 1px solid #e7e5e4;
                border-radius: 12px 12px 0 0;
                border-bottom: none;
                padding: 0.85rem 1.1rem;
                margin-top: 1.1rem; margin-bottom: 0;
                color: #0c0a09; font-weight: 700; font-size: 1.08rem;
            ">{icon} {dim_esc} — <span style="color:{dim_colors['header_accent']};">{score:.1f}/10</span>
            <span style="font-weight:500; color:#57534e;">({label_esc})</span>{skill_suffix}</div>
            """,
            unsafe_allow_html=True,
        )

        # 1. Overarching assessment (rationale) — shown first
        st.markdown(
            f"""
            <div style="
                background: #ffffff; border: 1px solid #e7e5e4;
                border-top: none;
                padding: 0.9rem 1.15rem 0.75rem 1.15rem;
                margin-bottom: 0;
            ">
                <div style="font-weight: 800; font-size: 0.68rem; letter-spacing: 0.07em;
                    text-transform: uppercase; color: #57534e; margin-bottom: 0.45rem;">Assessment</div>
                <p style="color:#44403c; line-height:1.65; margin:0; font-size:0.97rem;">
                    {html.escape(rationale).replace(chr(10), '<br/>')}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Evidence and Coach's suggestion — side by side
        col_left, col_right = st.columns(2)
        sq_esc = html.escape(str(student_quote)).replace("\n", "<br/>")
        if alt_phrasing:
            alt_esc = html.escape(str(alt_phrasing)).replace("\n", "<br/>")
            alt_body = alt_esc
        else:
            alt_body = (
                '<span style="color:#a8a29e;font-style:italic;">No alternative phrasing for this '
                "dimension—often because the evidence highlights a strength, not a closed turn.</span>"
            )

        quote_style = (
            "border-left: 4px solid #d97706; background: #fffbeb;"
            if score < 8.0
            else "border-left: 4px solid #059669; background: #ecfdf5;"
        )

        with col_left:
            st.markdown(
                f"""
                <div style="padding: 0; margin: 0;">
                  <div style="
                    {quote_style}
                    border-radius: 0 0 0 0;
                    padding: 1rem 1.15rem;
                    min-height: 9rem;
                    border: 1px solid #e7e5e4;
                    border-top: none;
                    border-right: none;
                  ">
                    <div style="font-weight: 800; font-size: 0.68rem; letter-spacing: 0.07em;
                        text-transform: uppercase; color: #57534e; margin-bottom: 0.55rem;">What you said</div>
                    <div style="font-family: Georgia, 'Times New Roman', serif; font-size: 1.02rem;
                        color: #1c1917; line-height: 1.6;">{sq_esc}</div>
                    <div style="margin-top: 0.85rem; font-size: 0.75rem; color: #78716c;">Turn {html.escape(str(turn_id))}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_right:
            suggest_border = "#fbbf24" if score < 8.0 else "#34d399"
            suggest_bg = "#fffbeb" if score < 8.0 else "#ecfdf5"
            st.markdown(
                f"""
                <div style="padding: 0; margin: 0;">
                  <div style="
                    background: {suggest_bg};
                    border: 1px solid #e7e5e4;
                    border-top: none;
                    border-left: 1px solid {suggest_border};
                    border-radius: 0 0 0 0;
                    padding: 1rem 1.15rem;
                    min-height: 9rem;
                  ">
                    <div style="font-weight: 800; font-size: 0.68rem; letter-spacing: 0.07em;
                        text-transform: uppercase; color: #57534e; margin-bottom: 0.55rem;">Coach's suggestion</div>
                    <div style="color: #1c1917; line-height: 1.6; font-size: 0.98rem;">{alt_body}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 3. Consequences — stakeholder cue + missed insight (only for lower-scoring turns)
        missed_plain = missed_insight_text(res)
        insight_html = dimension_insight_card_html(
            escape_html_multiline(str(stakeholder_cue)),
            escape_html_multiline(missed_plain),
            show_missed_insight=score < 9.0,
        )
        st.markdown(insight_html, unsafe_allow_html=True)

    actions = tactical_game_plan_three(evaluation_data)
    bullets_html = "".join(
        f"<li style='margin:0.5rem 0; color:#292524; line-height:1.55;'>{html.escape(a)}</li>" for a in actions
    )
    plan_colors = coach_accent_colors(session_skill_badge_parts(results, meta)[0] if results else 5.0)
    st.markdown(
        f"""
        <div style="
            background: {plan_colors['badge_soft']};
            border: 1px solid #e7e5e4;
            border-left: 5px solid {plan_colors['header_border']};
            border-radius: 14px;
            padding: 1.35rem 1.5rem;
            margin: 1.5rem 0 1.25rem 0;
            box-shadow: 0 4px 14px rgba(28, 25, 23, 0.06);
        ">
            <h3 style="color:#0c0a09; font-weight: 700; margin: 0 0 0.5rem 0; font-size: 1.2rem;">Tactical game plan</h3>
            <p style="color:#57534e; font-size: 0.92rem; margin: 0 0 0.75rem 0;">Your next session — three concrete habits to practice.</p>
            <ul style="margin: 0; padding-left: 1.25rem;">{bullets_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Conversation transcript — collapsible
    with st.expander("Conversation transcript", expanded=False):
        st.markdown(
            "<p style='color:#78716c; font-size:0.9rem; margin:0 0 0.75rem 0;'>Full dialogue from the selected interview.</p>",
            unsafe_allow_html=True,
        )
        turns = transcript_data.get("turns") or []
        parts = [
            "<div style='background:#fff; border:1px solid #e7e5e4; border-radius:12px; padding:1.25rem;'>"
        ]
        for t in turns:
            tid = t.get("turn_id", "")
            speaker = html.escape(str(t.get("speaker", "?")))
            text = html.escape(str(t.get("text", ""))).replace("\n", "<br/>")
            parts.append(
                f"<p style='color:#292524; font-weight:600; margin:0.75rem 0 0.25rem 0;'>Turn {tid} — {speaker}</p>"
            )
            parts.append(
                f"<p style='margin-left:0.75rem; padding:0.65rem 0.85rem; border-left:3px solid #d6d3d1; "
                f"background:#fafaf9; border-radius:0 8px 8px 0; color:#44403c; line-height:1.55;'>{text}</p>"
            )
        parts.append("</div>")
        st.markdown("".join(parts), unsafe_allow_html=True)

    # Glossary — definitions for technical interview terms used in feedback
    with st.expander("Glossary — interview technique terms", expanded=False):
        st.markdown(
            "<p style='color:#78716c; font-size:0.9rem; margin:0 0 0.75rem 0;'>"
            "Definitions for common terms that may appear in coach feedback above.</p>",
            unsafe_allow_html=True,
        )
        glossary_parts: List[str] = []
        for i, entry in enumerate(GLOSSARY):
            divider = "border-top: 1px solid #e7e5e4; margin-top: 0.85rem; padding-top: 0.85rem;" if i > 0 else ""
            glossary_parts.append(
                f'<div style="{divider}">'
                f'<span style="font-weight:700; color:#292524; font-size:0.97rem;">{html.escape(entry["term"])}</span>'
                f'<p style="color:#44403c; line-height:1.65; margin:0.3rem 0 0 0; font-size:0.93rem;">'
                f'{html.escape(entry["definition"])}</p>'
                f"</div>"
            )
        st.markdown(
            "<div style='background:#fff; border:1px solid #e7e5e4; border-radius:12px; padding:1.25rem;'>"
            + "".join(glossary_parts)
            + "</div>",
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")
    st.sidebar.caption("Interview Diagnostic Coach · IQR")


if __name__ == "__main__":
    main()
