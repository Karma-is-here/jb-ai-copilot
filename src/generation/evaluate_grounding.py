import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


GENERATION_FILE = Path("data/benchmarks/generation_evaluation.json")
BENCHMARK_DIR = Path("data/benchmarks")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def normalize_document_id(benchmark_file):
    """
    Convert:

    investment/advisory/advisory-mandates_questions.json

    into:

    investment/advisory/advisory-mandates
    """
    path = Path(benchmark_file)

    relative = path.relative_to(BENCHMARK_DIR)
    name = relative.stem

    if name.endswith("_questions"):
        name = name[:-len("_questions")]

    return str(Path(relative.parent) / name).replace("\\", "/")


def extract_citations(answer):
    """
    Extract [1], [2], [SOURCE 1], 【1】, 【SOURCE 1】 etc.
    """
    if not answer:
        return []

    patterns = [
        r"\[(?:SOURCE\s*)?(\d+)\]",
        r"【(?:SOURCE\s*)?(\d+)】",
    ]

    citations = []

    for pattern in patterns:
        citations.extend(int(x) for x in re.findall(pattern, answer, re.I))

    return sorted(set(citations))


def source_is_correct(source, expected_document, expected_page):
    if not source:
        return False

    document_match = (
        source.get("document_id") == expected_document
    )

    page_start = source.get("page_start")
    page_end = source.get("page_end")

    if page_start is None or page_end is None:
        return False

    page_match = page_start <= expected_page <= page_end

    return document_match and page_match


def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def main():
    with open(GENERATION_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]

    print("=" * 80)
    print("GROUNDING + CITATION TARGET EVALUATION")
    print("=" * 80)
    print(f"Generation results : {len(results)}")
    print()

    # ---------------------------------------------------------
    # Load embedding model locally.
    # No LLM/API calls.
    # ---------------------------------------------------------

    print("Loading local embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    generated_answers = []
    reference_answers = []

    detailed_results = []

    total_citations = 0
    correct_citations = 0

    answers_with_citation = 0
    answers_with_correct_citation = 0

    for result in results:

        question = result.get("question", "")
        answer = result.get("answer", "") or result.get("raw_answer", "")

        expected_answer = result.get("expected_answer", "")
        expected_page = result.get("expected_page")
        benchmark_file = result.get("benchmark_file", "")

        expected_document = normalize_document_id(benchmark_file)

        citations = extract_citations(answer)

        sources = result.get("sources", {})

        # Handle either {"1": {...}} or {1: {...}}
        normalized_sources = {
            str(k): v for k, v in sources.items()
        }

        correct_for_answer = 0
        citation_details = []

        for citation_id in citations:

            source = normalized_sources.get(str(citation_id))

            is_correct = source_is_correct(
                source,
                expected_document,
                expected_page,
            )

            total_citations += 1

            if is_correct:
                correct_citations += 1
                correct_for_answer += 1

            citation_details.append({
                "citation": citation_id,
                "correct_target": is_correct,
                "document_id": (
                    source.get("document_id")
                    if source else None
                ),
                "page_start": (
                    source.get("page_start")
                    if source else None
                ),
                "page_end": (
                    source.get("page_end")
                    if source else None
                ),
            })

        if citations:
            answers_with_citation += 1

        if correct_for_answer > 0:
            answers_with_correct_citation += 1

        detailed_results.append({
            "question": question,
            "benchmark_file": benchmark_file,
            "expected_document": expected_document,
            "expected_page": expected_page,
            "answer": answer,
            "expected_answer": expected_answer,
            "citations": citations,
            "correct_citation_count": correct_for_answer,
            "citation_details": citation_details,
        })

        generated_answers.append(answer)
        reference_answers.append(expected_answer)

    # ---------------------------------------------------------
    # Semantic similarity
    # ---------------------------------------------------------

    print("Computing offline reference-answer similarity...")

    generated_embeddings = model.encode(
        generated_answers,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    reference_embeddings = model.encode(
        reference_answers,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    similarities = [
        float(np.dot(g, r))
        for g, r in zip(
            generated_embeddings,
            reference_embeddings,
        )
    ]

    for item, similarity in zip(detailed_results, similarities):
        item["reference_answer_similarity"] = similarity

    # ---------------------------------------------------------
    # Aggregate metrics
    # ---------------------------------------------------------

    n = len(results)

    citation_target_precision = (
        correct_citations / total_citations
        if total_citations
        else 0.0
    )

    correct_citation_coverage = (
        answers_with_correct_citation / n
        if n
        else 0.0
    )

    citation_presence = (
        answers_with_citation / n
        if n
        else 0.0
    )

    similarities_np = np.array(similarities)

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"Questions                         : {n}")
    print(f"Answers with citations             : {answers_with_citation}")
    print(f"Citation presence                  : {citation_presence:.3f}")

    print()
    print(f"Total citations                    : {total_citations}")
    print(f"Correct citation targets           : {correct_citations}")
    print(
        f"Citation target precision          : "
        f"{citation_target_precision:.3f}"
    )

    print()
    print(
        f"Answers with ≥1 correct citation   : "
        f"{answers_with_correct_citation}"
    )
    print(
        f"Correct citation coverage          : "
        f"{correct_citation_coverage:.3f}"
    )

    print()
    print(
        f"Mean reference-answer similarity   : "
        f"{similarities_np.mean():.3f}"
    )

    print(
        f"Median reference-answer similarity : "
        f"{np.median(similarities_np):.3f}"
    )

    print(
        f"P10 similarity                     : "
        f"{np.percentile(similarities_np, 10):.3f}"
    )

    print(
        f"P90 similarity                     : "
        f"{np.percentile(similarities_np, 90):.3f}"
    )

    # Diagnostic thresholds only.
    for threshold in [0.60, 0.70, 0.80, 0.90]:
        rate = float(np.mean(similarities_np >= threshold))
        print(
            f"Similarity >= {threshold:.2f}              : "
            f"{rate:.3f}"
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output = {
        "summary": {
            "questions": n,
            "answers_with_citations": answers_with_citation,
            "citation_presence": citation_presence,
            "total_citations": total_citations,
            "correct_citations": correct_citations,
            "citation_target_precision": citation_target_precision,
            "answers_with_correct_citation": answers_with_correct_citation,
            "correct_citation_coverage": correct_citation_coverage,
            "mean_reference_answer_similarity": float(
                similarities_np.mean()
            ),
            "median_reference_answer_similarity": float(
                np.median(similarities_np)
            ),
        },
        "results": detailed_results,
    }

    output_file = Path(
        "data/benchmarks/grounding_evaluation.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print(f"Saved: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()