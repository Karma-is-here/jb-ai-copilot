import json
import re
import unicodedata
from pathlib import Path


FILE = Path("data/benchmarks/generation_evaluation.json")


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in",
    "on", "for", "with", "is", "was", "were", "are",
    "be", "by", "as", "at", "from", "that", "this",
    "it", "its", "their", "they", "he", "she", "you",
}


ABSTENTION_PATTERNS = [
    r"available sources do not",
    r"available sources.*not provide",
    r"do not provide enough information",
    r"not enough information",
    r"cannot determine",
    r"can't determine",
    r"unable to determine",
    r"unable to answer",
    r"cannot answer",
    r"can't answer",
    r"not specified",
]


def normalize(text):
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()

    # Normalize common punctuation variations.
    text = text.replace("-", "-")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("’", "'")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokens(text):
    text = normalize(text)

    words = re.findall(
        r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*",
        text,
    )

    return [
        w for w in words
        if w not in STOPWORDS
    ]


def is_abstention(answer):
    answer = normalize(answer)

    return any(
        re.search(pattern, answer)
        for pattern in ABSTENTION_PATTERNS
    )


def extract_numbers(text):
    """
    Extract numbers, percentages, dates, decimals, etc.
    """
    text = normalize(text)

    return re.findall(
        r"\b\d+(?:[.,]\d+)?%?\b",
        text,
    )


def exact_match(expected, answer):
    return normalize(expected) in normalize(answer)


def token_recall(expected, answer):
    expected_tokens = tokens(expected)
    answer_tokens = set(tokens(answer))

    if not expected_tokens:
        return 0.0

    matched = sum(
        token in answer_tokens
        for token in expected_tokens
    )

    return matched / len(expected_tokens)


def main():

    with open(FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]

    total = len(results)

    exact = 0
    high_token_recall = 0
    numeric_match = 0

    abstentions = 0

    detailed = []

    for r in results:

        expected = r.get("expected_answer", "")
        answer = r.get("answer", "")

        exact_hit = exact_match(
            expected,
            answer,
        )

        recall = token_recall(
            expected,
            answer,
        )

        expected_numbers = extract_numbers(expected)
        answer_numbers = extract_numbers(answer)

        if expected_numbers:
            numbers_ok = all(
                number in answer_numbers
                for number in expected_numbers
            )
        else:
            numbers_ok = True

        abstained = is_abstention(answer)

        if exact_hit:
            exact += 1

        if recall >= 0.8:
            high_token_recall += 1

        if numbers_ok and expected_numbers:
            numeric_match += 1

        if abstained:
            abstentions += 1

        detailed.append({
            "question": r["question"],
            "expected_answer": expected,
            "answer": answer,
            "exact_match": exact_hit,
            "token_recall": recall,
            "expected_numbers": expected_numbers,
            "answer_numbers": answer_numbers,
            "numeric_match": numbers_ok,
            "abstention": abstained,
        })

    print("=" * 80)
    print("OFFLINE ANSWER MATCH ANALYSIS")
    print("=" * 80)

    print(f"Questions                         : {total}")

    print(
        f"Exact/normalized answer present  : "
        f"{exact} ({exact / total:.3f})"
    )

    print(
        f"Token recall >= 0.80             : "
        f"{high_token_recall} "
        f"({high_token_recall / total:.3f})"
    )

    print(
        f"Answers with expected numbers    : "
        f"{numeric_match}"
    )

    print(
        f"Abstentions                       : "
        f"{abstentions} "
        f"({abstentions / total:.3f})"
    )

    print()
    print("LOW TOKEN-RECALL CASES")
    print("=" * 80)

    failures = sorted(
        detailed,
        key=lambda x: x["token_recall"],
    )

    for item in failures[:20]:

        print()
        print(
            f"RECALL: {item['token_recall']:.3f}"
        )

        print(
            f"Q: {item['question']}"
        )

        print(
            f"EXPECTED: {item['expected_answer']}"
        )

        print(
            f"ANSWER: {item['answer']}"
        )

        print(
            f"ABSTENTION: {item['abstention']}"
        )

    output = Path(
        "data/benchmarks/answer_match_evaluation.json"
    )

    with open(output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "questions": total,
                    "exact_match": exact / total,
                    "token_recall_ge_0_80":
                        high_token_recall / total,
                    "abstention_rate":
                        abstentions / total,
                },
                "results": detailed,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()