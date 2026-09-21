# src/generation/evaluate_generation.py

import json
from pathlib import Path

from src.generation.pipeline import RAGPipeline


BENCHMARK_DIR = Path("data/benchmarks")
OUTPUT_FILE = BENCHMARK_DIR / "generation_evaluation.json"


def load_questions():
    questions = []

    for path in sorted(BENCHMARK_DIR.rglob("*_questions.json")):
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)

        for item in items:
            questions.append(
                {
                    "benchmark_file": str(path),
                    "question": item["question"],
                    "expected_answer": item.get("answer"),
                    "expected_page": item.get("page"),
                }
            )

    return questions


def question_key(item):
    """
    Stable identifier used for checkpoint/resume.
    """
    return (
        item["benchmark_file"],
        item["question"],
    )


def load_checkpoint():
    """
    Load previously completed evaluation results.

    Returns:
        dict mapping question_key -> result
    """
    if not OUTPUT_FILE.exists():
        return {}

    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        print("Warning: existing evaluation file could not be read.")
        print("Starting a fresh evaluation.")
        return {}

    results = data.get("results", [])

    checkpoint = {}

    for result in results:
        # Only successful generations are considered completed.
        if "error" not in result:
            key = (
                result.get("benchmark_file"),
                result.get("question"),
            )

            if key[0] and key[1]:
                checkpoint[key] = result

    return checkpoint


def save_checkpoint(results, total_questions, stopped_reason=None):
    """
    Persist progress after every question.
    """
    successful = [
        r for r in results
        if "error" not in r
    ]

    errors = [
        r for r in results
        if "error" in r
    ]

    citation_valid = [
        r for r in successful
        if r["citation_validation"]["valid"]
    ]

    citation_present = [
        r for r in successful
        if r["citation_validation"]["citations_found"] > 0
    ]

    summary = {
        "questions": total_questions,
        "completed": len(successful),
        "successful": len(successful),
        "errors": len(errors),
        "citation_present": len(citation_present),
        "citation_valid": len(citation_valid),
        "citation_presence_rate": (
            len(citation_present) / len(successful)
            if successful
            else 0
        ),
        "citation_validity_rate": (
            len(citation_valid) / len(successful)
            if successful
            else 0
        ),
        "stopped_reason": stopped_reason,
    }

    output = {
        "summary": summary,
        "results": results,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )


def is_inference_error(error_message):
    """
    Identify errors where continuing would waste inference requests.
    """
    error_lower = error_message.lower()

    stop_patterns = [
        "402",
        "payment required",
        "depleted your monthly included credits",
        "quota",
        "rate limit",
        "429",
        "401 unauthorized",
        "403 forbidden",
        "invalid api key",
        "authentication",
    ]

    return any(
        pattern in error_lower
        for pattern in stop_patterns
    )


def evaluate():
    questions = load_questions()

    print("=" * 80)
    print("GENERATION EVALUATION")
    print("=" * 80)
    print(f"Questions in benchmark: {len(questions)}")

    checkpoint = load_checkpoint()

    completed_keys = set(checkpoint.keys())

    pending_questions = [
        item
        for item in questions
        if question_key(item) not in completed_keys
    ]

    print(f"Already completed:       {len(completed_keys)}")
    print(f"Remaining:               {len(pending_questions)}")
    print()

    if not pending_questions:
        print("All benchmark questions are already completed.")
        return

    # Start with previously successful results.
    results = list(checkpoint.values())

    print("Initializing RAG pipeline...")

    pipeline = RAGPipeline(
        candidate_k=20,
        context_k=5,
    )

    for index, item in enumerate(
        pending_questions,
        start=1,
    ):
        question = item["question"]

        print(
            f"[{index}/{len(pending_questions)}] "
            f"{question[:100]}"
        )

        try:
            result = pipeline.answer(question)

            validation = result["citation_validation"]

            evaluation_result = {
                "question": question,
                "expected_answer": item["expected_answer"],
                "expected_page": item["expected_page"],
                "benchmark_file": item["benchmark_file"],
                "answer": result["answer"],
                "raw_answer": result["raw_answer"],
                "citation_validation": validation,
                "sources": result["sources"],
                "retrieved_chunks": result["retrieved_chunks"],
            }

            results.append(evaluation_result)

            # SAVE IMMEDIATELY.
            save_checkpoint(
                results,
                total_questions=len(questions),
                stopped_reason=None,
            )

            print("  Saved.")

        except Exception as exc:
            error_message = str(exc)

            print(f"  ERROR: {error_message}")

            error_result = {
                "question": question,
                "expected_answer": item["expected_answer"],
                "expected_page": item["expected_page"],
                "benchmark_file": item["benchmark_file"],
                "error": error_message,
            }

            results.append(error_result)

            # Save the failure immediately.
            save_checkpoint(
                results,
                total_questions=len(questions),
                stopped_reason=(
                    error_message
                    if is_inference_error(error_message)
                    else None
                ),
            )

            # DO NOT waste further inference requests.
            if is_inference_error(error_message):
                print()
                print("=" * 80)
                print("INFERENCE STOPPED")
                print("=" * 80)
                print(
                    "An inference/quota/authentication error was detected."
                )
                print(
                    "Progress has been saved."
                )
                print(
                    f"Completed successfully: "
                    f"{len([r for r in results if 'error' not in r])}"
                )
                print(
                    f"Remaining questions: "
                    f"{len(questions) - len([r for r in results if 'error' not in r])}"
                )
                print()
                print(f"Saved to: {OUTPUT_FILE}")

                return

            # For non-inference errors, continue to the next question.
            continue

    # Final checkpoint.
    save_checkpoint(
        results,
        total_questions=len(questions),
        stopped_reason=None,
    )

    successful = [
        r for r in results
        if "error" not in r
    ]

    citation_valid = [
        r for r in successful
        if r["citation_validation"]["valid"]
    ]

    citation_present = [
        r for r in successful
        if r["citation_validation"]["citations_found"] > 0
    ]

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"Questions in benchmark: {len(questions)}"
    )

    print(
        f"Successful generations:  {len(successful)}"
    )

    print(
        f"Errors:                  "
        f"{len(results) - len(successful)}"
    )

    print(
        f"Citation presence:       "
        f"{len(citation_present) / len(successful):.3f}"
        if successful
        else "Citation presence:       N/A"
    )

    print(
        f"Citation validity:       "
        f"{len(citation_valid) / len(successful):.3f}"
        if successful
        else "Citation validity:       N/A"
    )

    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate()