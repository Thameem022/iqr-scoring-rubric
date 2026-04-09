from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Turn(BaseModel):
    """
    Represents a single dialogue exchange in an Intervista session.

    Each turn captures who is speaking, what was said, and any
    downstream analytic annotations (e.g., intent classification
    and IQR marker tags) used by the Intervista analytics pipeline.
    """

    turn_id: int = Field(..., description="Monotonically increasing 1-based index for the turn.")
    speaker: str = Field(..., description="Role label of the speaker (e.g., 'student', 'stakeholder').")
    text: str = Field(..., description="Raw text of the utterance for this turn.")
    intent: Optional[str] = Field(
        default=None,
        description="Optional downstream intent label assigned by classifiers or LLMs.",
    )
    iqr_markers: List[str] = Field(
        default_factory=list,
        description="List of rubric-relevant marker tags identified in this turn.",
    )

    @field_validator("turn_id")
    @classmethod
    def validate_turn_id_positive(cls, value: int) -> int:
        """
        Ensure that the turn identifier is always a positive integer.
        """
        if value <= 0:
            raise ValueError("turn_id must be a positive integer.")
        return value


class Transcript(BaseModel):
    """
    Container for an entire Intervista conversation transcript.

    The transcript aggregates per-turn data and arbitrary metadata
    (e.g., scenario identifiers, timestamps, cohort information) that
    downstream IQR and SIC evaluators consume.
    """

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata about the session (scenario, cohort, timestamps, etc.).",
    )
    turns: List[Turn] = Field(
        default_factory=list,
        description="Ordered list of dialogue turns that make up the transcript.",
    )


class Evidence(BaseModel):
    """
    Structured evidence supporting a single rubric dimension score.

    This links an IQR/SIC decision back to specific transcript
    content, including the turn where the behavior occurred, a
    salient student quote, the stakeholder's cue or context, and
    optional verbatim phrasing feedback for a stronger question.
    """

    turn_id: int = Field(..., description="Turn identifier where key evidence was observed.")
    student_quote: str = Field(
        ...,
        description="Short excerpt of the student's utterance motivating the score.",
    )
    stakeholder_cue: str = Field(
        ...,
        description="Relevant stakeholder statement, reaction, or contextual cue.",
    )
    alternative_phrasing: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim suggestion for a clearer or stronger interview question. "
            "Displayed beside the student quote as 'A better way to ask' in the dashboard."
        ),
    )

    @field_validator("turn_id")
    @classmethod
    def validate_turn_id_positive(cls, value: int) -> int:
        """
        Ensure that the evidence turn identifier is always a positive integer.
        """
        if value <= 0:
            raise ValueError("evidence.turn_id must be a positive integer.")
        return value


class IQREvaluation(BaseModel):
    """
    Encodes the score for a single IQR or SIC rubric dimension.

    Each instance represents one dimension (e.g., empathy, technical
    accuracy, stakeholder coverage) with a 10-point score (including
    half steps), a skill-level title, a label, a free-text rationale,
    optional line-of-inquiry impact, and structured evidence referencing
    the underlying transcript.
    """

    dimension_id: str = Field(
        ...,
        description="Stable identifier for the rubric dimension (e.g., 'IQ-01').",
    )
    dimension_name: str = Field(
        ...,
        description="Human-readable name of the rubric dimension.",
    )
    score: float = Field(
        ...,
        description="Rubric score on a 10-point scale (e.g., 6.5 for 6.5/10).",
    )
    skill_level_title: str = Field(
        ...,
        description=(
            "Descriptive skill indicator (e.g., 'Competent Operational Interviewer'). "
            "Shown on the diagnostic dashboard per dimension and aggregated in the session badge."
        ),
    )
    label: str = Field(
        ...,
        description="Textual label or band associated with the score.",
    )
    rationale: str = Field(
        ...,
        description="Natural language explanation justifying the assigned score.",
    )
    line_of_inquiry_impact: Optional[str] = Field(
        default=None,
        description=(
            "Type of insight (technical, ethical, contextual, etc.) that was "
            "inaccessible due to the student's phrasing. Surfaced in the dashboard "
            "as an amber 'insight loss' callout when present."
        ),
    )
    evidence: Evidence = Field(
        ...,
        description="Structured evidence pointing back to the source transcript turns.",
    )

    @field_validator("score")
    @classmethod
    def validate_score_range(cls, value: float) -> float:
        """
        Ensure that rubric scores respect the 10-point scale.
        """
        if not (1.0 <= value <= 10.0):
            raise ValueError("score must be between 1.0 and 10.0 (inclusive).")
        return value


class SessionEvaluation(BaseModel):
    """
    Final evaluated representation of an Intervista session.

    This model is the main output of the analytics pipeline,
    combining session-level metadata, per-dimension IQR/SIC
    evaluations, and a global narrative summary suitable for
    storage, dashboards, or downstream analytics.
    """

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Session-level metadata: input transcript fields plus evaluator-enriched "
            "keys such as iqr_score_scale, iqr_score_min, iqr_score_max, "
            "iqr_skill_bands for the 10-point diagnostic scale."
        ),
    )
    evaluation_results: List[IQREvaluation] = Field(
        default_factory=list,
        description="List of rubric dimension evaluations (IQR and SIC).",
    )
    overall_summary: str = Field(
        ...,
        description="Narrative, human-readable summary of overall session quality.",
    )

