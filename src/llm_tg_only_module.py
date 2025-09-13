import os
import json
import random
from collections import defaultdict
from openai import OpenAI
from datasets import load_dataset

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def load_system_prompt():
    with open("src/prompts/system.txt", "r") as f:
        return f.read()


def run_instance_tg_only(instance, system_prompt=load_system_prompt(), temperature=0):
    """Run one dataset instance using only the temporal graph (TG) + question (no ASP facts, no story)."""

    # --- Build prompt ---
    tg_str = "\n".join(instance.get("TG", []))
    candidates_str = "\n".join(instance["candidates"])

    question_prompt = f"""
You are given a temporal graph (TG) derived from a story, along with a question.
Answer the question by selecting exactly one of the candidate answers.

Temporal Graph (TG):
{tg_str}

Question:
{instance['question']}

Candidates:
{candidates_str}

Instructions:
- Use only the TG to support your reasoning.
- Select exactly one candidate answer.
- The value of "answer_choice" must exactly match one of the candidate answers.
- Return only JSON in the following format:

{{
  "reasoning": "<step-by-step explanation>",
  "answer_choice": "<the candidate answer you selected>"
}}
"""

    # --- Query model ---
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question_prompt},
        ],
        temperature=temperature,
    )
    resp_text = resp.choices[0].message.content
    try:
        answer_choice = json.loads(resp_text)["answer_choice"]
    except Exception:
        raise ValueError(f"TG-only output not valid JSON: {resp_text}")

    # --- Check correctness ---
    gold_answer = instance["answer"]
    match = answer_choice in gold_answer

    return {
        "system_prompt": system_prompt,
        "question_prompt": question_prompt,
        "response": resp_text,
        "answer_choice": answer_choice,

        # Ground truth
        "gold_answer": gold_answer,
        "match": match,

        # Instance metadata
        "instance_TG": tg_str,
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
        buckets[inst["Q-Type"]].append(inst)

    n_types = len(buckets)
    per_type = n // n_types

    sampled = []
    for qtype, items in buckets.items():
        if items:
            sampled.extend(random.sample(items, min(per_type, len(items))))

    return sampled


# -----------------------------
# Batch runner
# -----------------------------

def run_batch_tg_only(n=50, mode="random", output_path="results/llm_results_tg_only.json"):
    dataset = load_dataset("sxiong/TGQA", "TGQA_TGR")["test"]

    if mode == "random":
        subset = sample_random(dataset, n)
    elif mode == "stratified":
        subset = sample_stratified(dataset, n)
    else:
        raise ValueError("mode must be 'random' or 'stratified'")

    results = {}
    for i, instance in enumerate(subset):
        print(f"[TG-only] Processing {instance['id']} ({i+1}/{len(subset)})...")
        try:
            results[instance["id"]] = run_instance_tg_only(instance)
        except Exception as e:
            results[instance["id"]] = {"error": str(e)}

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved TG-only results for {len(subset)} instances to {output_path}")


if __name__ == "__main__":
    # Example: run 50 stratified samples
    run_batch_tg_only(n=50, mode="stratified")