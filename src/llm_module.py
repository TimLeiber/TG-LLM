import os
import json
import random
from collections import defaultdict
from openai import OpenAI
from datasets import load_dataset
from prompt_generation import make_question_prompt, query_asp_output_prompt


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-3.5-turbo"
DATA = 'TimeQA_TGR'

# Load ASP results (facts per story)
with open("results/asp_results.json") as f:
    asp_results = json.load(f)


def get_story_key(instance_id: str) -> str:
    """
    Map dataset id like '/wiki/Huw_Irranca-Davies#P39_hard_5'
    -> '_wiki_Huw_Irranca-Davies#P39.lp'
    """
    # strip "_hard" and trailing question index
    if "_hard" in instance_id:
        base = instance_id.split("_hard")[0]
    else:
        base = instance_id
    
    # replace leading '/' with '_' (and any other '/' if present)
    base = base.replace("/", "_")

    return f"{base}.lp"

def load_system_prompt():
    with open("src/prompts/system.txt", "r") as f:
        return f.read()


def run_instance(instance, system_prompt=load_system_prompt(), temperature=0):
    """Run one dataset instance through the two-step LLM pipeline."""

    # --- Stage 1: predicate choice ---
    query_prompt = query_asp_output_prompt(instance["question"])
    stage1_resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_prompt},
        ],
        temperature=temperature,
    )
    stage1_text = stage1_resp.choices[0].message.content
    try:
        stage1_json = json.loads(stage1_text)
        predicate_choice = stage1_json["predicate_choice"]
    except Exception:
        raise ValueError(f"Stage 1 output not valid JSON: {stage1_text}")

    # Normalize predicate_choice to always be a list
    if isinstance(predicate_choice, str):
        predicate_choice = [predicate_choice]

    # --- Stage 2: answer selection ---
    # --- Stage 2: answer selection ---
    story_key = get_story_key(instance["id"])
    if story_key not in asp_results:
        raise KeyError(f"{story_key} not found in asp_results.json")

    asp_facts = []
    for pred in predicate_choice:
        asp_facts.extend(asp_results[story_key].get(pred, []))

    candidates_str = "\n".join(instance["candidates"])
    events_str = "\n".join(instance["TG"]) if "TG" in instance else ""  # base events
    tg_str = "\n".join(instance["TG"]) if "TG" in instance else ""      # temporal graph

    question_prompt = make_question_prompt(
        instance["question"],
        asp_facts,
        candidates_str,
        events_str,
        tg_str  # NEW: pass TG explicitly
    )

    stage2_resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question_prompt},
        ],
        temperature=0,
    )
    stage2_text = stage2_resp.choices[0].message.content
    try:
        answer_choice = json.loads(stage2_text)["answer_choice"]
    except Exception:
        raise ValueError(f"Stage 2 output not valid JSON: {stage2_text}")

    # --- Check correctness ---
    gold_answer = instance["answer"]
    match = answer_choice in gold_answer

    return {
        # System + prompts + responses
        "system_prompt_stage1": system_prompt,
        "query_prompt": query_prompt,
        "stage1_response": stage1_text,
        "predicate_choice": predicate_choice,
        "system_prompt_stage2": system_prompt,
        "question_prompt": question_prompt,
        "stage2_response": stage2_text,
        "answer_choice": answer_choice,

        # Ground truth
        "gold_answer": gold_answer,
        "match": match,

        # Instance metadata
        "instance_question": instance["question"],
        "instance_candidates": instance["candidates"],
        "instance_gold_answer": instance["answer"],
        "instance_id": instance["id"]
    }


# -----------------------------
# Sampling functions
# -----------------------------

def sample_random(dataset, n=50, seed=42):
    random.seed(seed)
    return random.sample(list(dataset), n)


def sample_stratified(dataset, n=50, seed=42):
    random.seed(seed)
    buckets = defaultdict(list)
    for inst in dataset:
        qtype = inst.get("Q-Type") or "Unknown"
        buckets[qtype].append(inst)

    # if all Q-Types are "Unknown", just sample randomly
    if len(buckets) == 1 and "Unknown" in buckets:
        return sample_random(dataset, n, seed)

    n_types = len(buckets)
    per_type = max(1, n // n_types)

    sampled = []
    for qtype, items in buckets.items():
        if items:
            sampled.extend(random.sample(items, min(per_type, len(items))))

    return sampled
def sample_stratified(dataset, n=50, seed=42):
    random.seed(seed)
    buckets = defaultdict(list)
    for inst in dataset:
        qtype = inst.get("Q-Type") or "Unknown"
        buckets[qtype].append(inst)

    # if all Q-Types are "Unknown", just sample randomly
    if len(buckets) == 1 and "Unknown" in buckets:
        return sample_random(dataset, n, seed)

    n_types = len(buckets)
    per_type = max(1, n // n_types)

    sampled = []
    for qtype, items in buckets.items():
        if items:
            sampled.extend(random.sample(items, min(per_type, len(items))))

    return sampled


# -----------------------------
# Batch runner
# -----------------------------

def run_batch(n=50, mode="random", output_path=f"results/llm_results_{MODEL}_{DATA}.json", data=DATA):
    dataset = load_dataset("sxiong/TGQA", data)["test" if data == 'TGQA_TGR' else 'hard_test']

    if mode == "random":
        subset = sample_random(dataset, n)
    elif mode == "stratified":
        subset = sample_stratified(dataset, n)
    else:
        raise ValueError("mode must be 'random' or 'stratified'")

    results = {}
    for i, instance in enumerate(subset):
        print(f"Processing {instance['id']} ({i+1}/{len(subset)})...")
        try:
            results[instance["id"]] = run_instance(instance)
        except Exception as e:
            results[instance["id"]] = {"error": str(e)}

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f" Saved results for {len(subset)} instances to {output_path}")


if __name__ == "__main__":
    # Example: run 50 stratified samples
    run_batch(n=500, mode="stratified")