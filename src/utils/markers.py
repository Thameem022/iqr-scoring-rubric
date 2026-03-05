from enum import Enum

class IQRMarker(str, Enum):
    """
    Linguistic and behavioral markers for the Interview Quality Rubric (IQR).
    These serve as evidence-based indicators for automated scoring[cite: 20, 127].
    """
    
    OPEN_ENDED_Q = "open_ended_question"
    CLOSED_ENDED_Q = "closed_ended_question"
    LEADING_Q = "leading_question_detected"
    DOUBLE_BARRELED_Q = "double_barreled_question"
    NEUTRAL_FRAMING = "neutral_framing"

    CLARIFICATION_REQ = "clarification_request"
    ELABORATION_PROBE = "elaboration_probe"
    COUNTER_QUESTION = "counter_question"
    TECHNICAL_FACT_FOLLOWUP = "specific_technical_followup"
    DEPTH_CHECK = "depth_check_on_impacts"

    REFLECTIVE_SUMMARY = "reflective_summary_of_concerns"
    VERBAL_ACK = "verbal_acknowledgment"
    UNDERSTANDING_CHECK = "checking_for_understanding"
    INTERRUPTION = "student_interruption_detected"

    EMOTIONAL_VALIDATION = "emotional_cue_validation"
    PERSPECTIVE_TAKING = "perspective_taking_statement"
    RESPECTFUL_TONE = "respectful_professional_tone"
    ADAPTIVE_PACING = "adaptive_conversational_pacing"

    PURPOSE_TRANSPARENCY = "transparency_about_interview_purpose"
    NEUTRAL_CONFLICT_STANCE = "neutral_stance_on_stakeholder_conflict"
    BIAS_SAFEGUARD = "adherence_to_bias_safeguards"
    FORMAL_STEM_VOCAB = "formal_stem_persona_maintenance"

    @classmethod
    def get_by_dimension(cls, dimension_prefix: str):
        """Helper to filter markers by their category."""
        return [m for m in cls if m.name.startswith(dimension_prefix)]