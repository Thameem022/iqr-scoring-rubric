from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.core.llm_factory import get_llm
from src.utils.schema import SessionEvaluation, Transcript

# Session metadata merged into every successful evaluation so downstream systems
# know the rubric uses a 1.0–10.0 float scale with Phase-1 skill titles.
IQR_RUBRIC_METADATA: Dict[str, Any] = {
    "iqr_score_scale": "10-point",
    "iqr_score_min": 1.0,
    "iqr_score_max": 10.0,
    "iqr_skill_bands": (
        "6.0–6.9 Novice Fact-Finder; 7.0–7.9 Emerging Technical Interviewer; "
        "8.0–8.9 Competent Operational Interviewer; 9.0–9.9 Advanced Systems "
        "Interviewer; 10.0 Master Stakeholder Partner"
    ),
}

# Injected next to the system prompt so the model maps score ↔ skill_level_title
# consistently with the diagnostic bands (including sub-6.0 scores).
IQR_SKILL_SCALE_CONTEXT = """\
**10-point mapping (apply per dimension):**
- Set `score` to a float in [1.0, 10.0] (half steps allowed, e.g. 6.5).
- Set `skill_level_title` to the **exact quoted title** from Phase 1 whose band contains that score (e.g. 7.4 → "Emerging Technical Interviewer").
- For scores **below 6.0**, choose the closest Phase 1 title by meaning and say so in `rationale`, or use a concise developmental label that fits the band (e.g. approaching "Novice Fact-Finder").
- **Closed-turn rule:** If Phase 2 identifies a "Closed Turn" for evidence tied to a dimension, you **must** populate `evidence.alternative_phrasing` (the Bridge) and `line_of_inquiry_impact` (the lost line of inquiry). If there is no closed turn for that dimension’s evidence, set both fields to `null`.
"""


class IQRScorer:
    """
    Model-agnostic scorer for the Interview Quality Rubric (IQR).

    This class wraps a LangChain-compatible chat LLM and a system prompt to
    produce structured `SessionEvaluation` outputs from an input `Transcript`.
    It is intentionally provider-agnostic so that GPT-4o, Claude, and other
    models can be swapped via the `llm_factory` without changing scoring logic,
    which is critical for NSF reliability and IRR studies.
    """

    def __init__(self, llm: BaseChatModel, prompt_path: str) -> None:
        """
        Parameters
        ----------
        llm:
            A LangChain chat model instance (e.g., `ChatOpenAI`, `ChatAnthropic`)
            constructed by the `get_llm` factory.
        prompt_path:
            Filesystem path to the IQR system prompt that encodes the
            Stakeholder Cue → Student Response → Marker Tagging instructions.
        """
        self._llm = llm
        self._prompt_path = Path(prompt_path)
        self._system_prompt = self._load_prompt(self._prompt_path)

        # Pydantic-based structured output parser: must match `SessionEvaluation`
        # in src/utils/schema.py (nested IQREvaluation + Evidence diagnostic fields).
        self._parser = PydanticOutputParser(pydantic_object=SessionEvaluation)

        # Build the internal LangChain chain once and reuse it.
        self._chain = self._build_chain()

    @staticmethod
    def _load_prompt(path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(f"IQR system prompt not found at: {path}")
        return path.read_text(encoding="utf-8")

    def _build_chain(self, llm: Optional[BaseChatModel] = None):
        """
        Construct a LangChain pipeline that:
        1. Injects the IQR system prompt from disk.
        2. Provides minimal wrapper variables for format instructions and transcript JSON.
        3. Parses the final JSON into a `SessionEvaluation` instance.

        All qualitative grading and chain-of-thought logic lives in the external
        `iqr_system_prompt.txt` so rubric iterations do not require code changes.
        """

        active_llm = llm or self._llm

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "{system_prompt}\n\n"
                        "{rubric_context}\n\n"
                        "{format_instructions}"
                    ),
                ),
                (
                    "user",
                    (
                        "Evaluate the following interview transcript and return "
                        "a single JSON object matching the `SessionEvaluation` schema.\n\n"
                        "Transcript JSON:\n```json\n{transcript_json}\n```"
                    ),
                ),
            ]
        )

        return prompt | active_llm | self._parser

    async def evaluate(self, transcript: Transcript) -> SessionEvaluation:
        """
        Run the IQR evaluation pipeline on a transcript.

        The method is fully async and returns a `SessionEvaluation` instance
        so that downstream code can remain agnostic to the underlying LLM
        provider and model choice.
        """
        # Edge case: empty transcript should not be scored.
        if not transcript.turns:
            base_metadata = {**dict(transcript.metadata or {}), **IQR_RUBRIC_METADATA}
            base_metadata["status"] = "Incomplete"
            return SessionEvaluation(
                metadata=base_metadata,
                evaluation_results=[],
                overall_summary="Transcript is empty; no IQR evaluation was performed.",
            )

        # Extract and carry forward core session metadata for standardized output.
        base_metadata = dict(transcript.metadata or {})
        for key in ("session_id", "persona_id", "scenario_id"):
            # Ensure the keys exist, even if set to None, for downstream consumers.
            base_metadata.setdefault(key, base_metadata.get(key))

        transcript_payload: Any = transcript.model_dump()
        transcript_json = json.dumps(transcript_payload, ensure_ascii=False, indent=2)

        chain_input = {
            "system_prompt": self._system_prompt,
            "rubric_context": IQR_SKILL_SCALE_CONTEXT,
            "format_instructions": self._parser.get_format_instructions(),
            "transcript_json": transcript_json,
        }

        try:
            result: SessionEvaluation = await self._chain.ainvoke(chain_input)

            # Session + transcript metadata, then model-returned metadata, then fixed
            # IQR scale keys (so stakeholders always see 10-point mapping context).
            result.metadata = {
                **base_metadata,
                **result.metadata,
                **IQR_RUBRIC_METADATA,
            }
            return result
        except Exception:
            # Fallback logic: retry with a high-reliability OpenAI model (e.g., GPT-4o)
            # to satisfy FR-13 reliability requirements.
            fallback_llm = get_llm(provider="openai", model_name="gpt-4o", temperature=0.0)
            fallback_chain = self._build_chain(llm=fallback_llm)

            result: SessionEvaluation = await fallback_chain.ainvoke(chain_input)

            result.metadata = {
                **base_metadata,
                **result.metadata,
                **IQR_RUBRIC_METADATA,
            }
            return result


