import os
import json
import re
from collections import Counter

RESULTS_DIR = "results"

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

def evaluate_file(path: str):
    with open(path) as f:
        results = json.load(f)

    total = 0
    acc = 0
    em = 0
    f1 = 0.0

    for instance_id, res in results.items():
        if "error" in res or "answer_choice" not in res:
            continue  # skip failed or invalid cases
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

    if total == 0:
        return None

    return {
        "file": os.path.basename(path),
        "total": total,
        "acc": acc / total,
        "em": em / total,
        "f1": f1 / total,
    }

def evaluate_all(results_dir=RESULTS_DIR):
    files = [f for f in os.listdir(results_dir) if f.endswith(".json") and f != "asp_results.json"]

    all_results = []
    for fname in files:
        res = evaluate_file(os.path.join(results_dir, fname))
        if res:
            all_results.append(res)

    for r in all_results:
        print(f"\nFile: {r['file']}")
        print(f"  Evaluated {r['total']} instances")
        print(f"  Accuracy: {r['acc']*100:.5f}%")
        print(f"  Exact Match (EM): {r['em']*100:.5f}%")
        print(f"  F1: {r['f1']*100:.5f}%")

if __name__ == "__main__":
    evaluate_all()
