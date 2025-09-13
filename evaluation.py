import json
import re
from collections import Counter

RESULTS_PATH = "results/llm_results_tg_only.json"

def normalize_text(s: str) -> str:
    """Lowercase and remove punctuation/articles/extra spaces for EM/F1."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)  # remove articles
    s = re.sub(r'[^a-z0-9\s]', ' ', s)     # remove punctuation
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def f1_score(pred: str, gold: str) -> float:
    """Token-level F1 between two normalized strings."""
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def evaluate(results_path=RESULTS_PATH):
    with open(results_path) as f:
        results = json.load(f)

    total = 0
    acc = 0
    em = 0
    f1 = 0.0

    for instance_id, res in results.items():
        if "error" in res:
            continue  # skip failed cases
        total += 1

        pred = res["answer_choice"]
        gold_answers = res["gold_answer"]

        # Accuracy: exact candidate match (no normalization)
        if pred in gold_answers:
            acc += 1

        # EM / F1: normalized
        em_this = 0
        f1_this = 0
        for gold in gold_answers:
            if normalize_text(pred) == normalize_text(gold):
                em_this = 1
            f1_this = max(f1_this, f1_score(pred, gold))

        em += em_this
        f1 += f1_this

    acc = acc / total if total else 0
    em = em / total if total else 0
    f1 = f1 / total if total else 0

    print(f"Evaluated {total} instances")
    print(f"Accuracy: {acc:.3f}")
    print(f"Exact Match (EM): {em:.3f}")
    print(f"F1: {f1:.3f}")

if __name__ == "__main__":
    evaluate()