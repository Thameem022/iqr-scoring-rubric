#!/usr/bin/env python
"""
Phase 4.1 Inter-Rater Reliability (IRR) calibration script.

Runs IQRScorer across different transcripts and LLM providers, prints
dimension scores and rationales, highlights negative markers (e.g. in Bronze
anchors), and saves full SessionEvaluation JSON for downstream analysis.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Set

# Project root for imports and resolving paths
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.llm_factory import get_llm
from src.evaluators.iqr_scorer import IQRScorer
from src.utils.schema import SessionEvaluation, Transcript
from src.utils.markers import IQRMarker

# Markers that indicate problematic behavior (leading questions, interruptions, etc.)
NEGATIVE_MARKERS: Set[str] = {
    IQRMarker.LEADING_Q.value,
    IQRMarker.DOUBLE_BARRELED_Q.value,
    IQRMarker.INTERRUPTION.value,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run IQR calibration across transcripts and LLM providers (Phase 4.1 IRR)."
    )
    parser.add_argument(
        "--transcript",
        required=True,
        help="Path to transcript JSON (e.g. data/transcripts/bronze/1.json).",
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=["openai", "anthropic"],
        help="LLM provider.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g. gpt-4o, claude-3-5-sonnet).",
    )
    return parser.parse_args()


def load_transcript(path: Path) -> Transcript:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Transcript.model_validate(data)


def find_negative_markers(transcript: Transcript) -> List[str]:
    found: Set[str] = set()
    for turn in transcript.turns:
        for marker in turn.iqr_markers or []:
            if marker in NEGATIVE_MARKERS:
                found.add(marker)
    return sorted(found)


def print_results_table(session_eval: SessionEvaluation) -> None:
    """Print Dimension | Score | Rationale snippet as a clean ASCII table."""
    rows = []
    for ev in session_eval.evaluation_results:
        snippet = (ev.rationale or "").strip().replace("\n", " ")
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."
        rows.append((ev.dimension_name, str(ev.score), snippet))

    if not rows:
        print("\n(No dimension results to display.)\n")
        return

    headers = ("Dimension", "Score", "Rationale snippet")
    col_widths = [
        max(len(h), max(len(r[i]) for r in rows))
        for i, h in enumerate(headers)
    ]

    def fmt_row(cols: tuple) -> str:
        return " | ".join(c.ljust(w) for c, w in zip(cols, col_widths))

    sep = "-+-".join("-" * w for w in col_widths)
    print()
    print(fmt_row(headers))
    print(sep)
    for r in rows:
        print(fmt_row(r))
    print()


def build_output_path(model: str, transcript_path: Path) -> Path:
    """e.g. gpt4o_bronze_1_results.json under data/output/."""
    safe_model = model.replace("-", "").replace(":", "_").lower()
    stem = transcript_path.stem
    tier = transcript_path.parent.name  # bronze, silver, gold
    filename = f"{safe_model}_{tier}_{stem}_results.json"
    out_dir = PROJECT_ROOT / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


async def run() -> None:
    args = parse_args()

    transcript_path = (PROJECT_ROOT / args.transcript.strip()).resolve()
    if not transcript_path.is_file():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    transcript = load_transcript(transcript_path)
    is_bronze = "bronze" in transcript_path.parts

    # Empty transcript: no LLM call
    if not transcript.turns:
        print("Transcript has no turns; returning Incomplete SessionEvaluation.")
        incomplete = SessionEvaluation(
            metadata={**(transcript.metadata or {}), "status": "Incomplete"},
            evaluation_results=[],
            overall_summary="Transcript is empty; no IQR evaluation was performed.",
        )
        print_results_table(incomplete)
        print("Overall Summary:\n", incomplete.overall_summary)
        out_path = build_output_path(args.model, transcript_path)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(incomplete.model_dump(), f, ensure_ascii=False, indent=2)
        print(f"Saved: {out_path}")
        return

    # Initialize LLM and scorer
    llm = get_llm(provider=args.provider, model_name=args.model, temperature=0.0)
    prompt_path = PROJECT_ROOT / "src" / "evaluators" / "prompts" / "iqr_system_prompt.txt"
    scorer = IQRScorer(llm=llm, prompt_path=str(prompt_path))

    # Run evaluation
    session_eval = await scorer.evaluate(transcript)

    # Reporting: ASCII table
    print_results_table(session_eval)

    # Highlight negative markers (especially for Bronze)
    neg_found = find_negative_markers(transcript)
    if neg_found:
        if is_bronze:
            print("[BRONZE] Negative markers detected in transcript:")
        else:
            print("Negative markers detected in transcript:")
        for m in neg_found:
            print(f"  - {m}")
    else:
        print("No configured negative markers detected in transcript.")

    # Reliability check: Overall Summary (CoT visibility)
    print("\nOverall Summary (reliability / CoT check):\n")
    print(session_eval.overall_summary)
    print()

    # Save full SessionEvaluation JSON
    out_path = build_output_path(args.model, transcript_path)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(session_eval.model_dump(), f, ensure_ascii=False, indent=2)
    print(f"Full SessionEvaluation saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(run())
