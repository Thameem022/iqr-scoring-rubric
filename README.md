## Intervista IQR Scoring Engine

This repository implements the Intervista Interview Quality Rubric (IQR) scoring engine for STEM interview transcripts. It is designed for NSF-grade reliability studies, large-scale classroom deployments (e.g., ID2050), and rapid rubric iteration across project years.

### Calibration Script for IRR Studies

The primary entry point for Phase 4.1 Inter-Rater Reliability (IRR) studies is the calibration script:

- **Script**: `tests/scripts/run_calibration.py`
- **Purpose**: Run the `IQRScorer` against different transcripts and LLM providers to evaluate scoring behavior and reliability.

#### Arguments

The script uses `argparse` with the following flags:

- `--transcript`: Path to a transcript JSON file (e.g., `data/transcripts/bronze/1.json`).
- `--provider`: LLM provider (`openai` or `anthropic`).
- `--model`: Model name (e.g., `gpt-4o`, `claude-3-5-sonnet`).

#### What the Script Does

- Initializes an LLM via `src/core/llm_factory.get_llm`.
- Loads the IQR system prompt from `src/evaluators/prompts/iqr_system_prompt.txt`.
- Wraps the LLM with `IQRScorer` (`src/evaluators/iqr_scorer.py`).
- Loads the transcript JSON into the `Transcript` Pydantic model.
- Calls `await scorer.evaluate(transcript)` to obtain a `SessionEvaluation`.
- Prints a clean ASCII table:
  - **Dimension** | **Score** | **Rationale snippet**
- Highlights whether any "negative" markers (e.g., `leading_question_detected`) appear in the transcript (especially useful for Bronze anchors).
- Prints the **Overall Summary** from the `SessionEvaluation` to verify that the chain-of-thought–based reasoning in the prompt is grounded and coherent.
- Saves the full `SessionEvaluation` JSON to `data/output/` with a filename that encodes model and transcript information, e.g.:
  - `data/output/gpt4o_bronze_1_results.json`

#### Example Usage

From the project root:

```bash
python tests/scripts/run_calibration.py \
  --transcript data/transcripts/bronze/1.json \
  --provider openai \
  --model gpt-4o
```

This will:

- Run the Bronze anchor transcript through the selected LLM.
- Output a table of rubric dimension scores and rationales.
- Surface any negative markers present in the transcript.
- Persist a full JSON record of the `SessionEvaluation` for downstream IRR analysis and audits.

