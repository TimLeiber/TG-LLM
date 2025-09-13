import os
import json
import random
from collections import defaultdict
from openai import OpenAI
from datasets import load_dataset

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-3.5-turbo" # any openai model
DATA = "TimeQA_TGR"  # or "TimeQA_TGR"

def load_system_prompt():
    with open("src/prompts/system.txt", "r") as f:
        return f.read()


def run_instance_story_only(instance, system_prompt=load_system_prompt(), temperature=0):
    """Run one dataset instance using only the story text + question (no ASP facts)."""

    # --- Build prompt ---
    story_str = instance.get("story", "")
    candidates_str = "\n".join(instance["candidates"])

    question_prompt = f"""
You are given a story and a question. 
Answer the question by selecting exactly one of the candidate answers.

Story:
{story_str}

Question:
{instance['question']}

Candidates:
{candidates_str}

Instructions:
- Use only the story to support your reasoning.
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
        model=MODEL,
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
        raise ValueError(f"Story-only output not valid JSON: {resp_text}")

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
        "instance_story": story_str,
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


# -----------------------------
# Batch runner
# -----------------------------

def run_batch_story_only(n=50, mode="random", output_path=None, data=DATA):
    dataset = load_dataset("sxiong/TGQA", data)["test" if data == "TGQA_TGR" else "hard_test"]

    if mode == "random":
        subset = sample_random(dataset, n)
    elif mode == "stratified":
        subset = sample_stratified(dataset, n)
    else:
        raise ValueError("mode must be 'random' or 'stratified'")

    if output_path is None:
        output_path = f"results/llm_results_story_only_{MODEL}_{data}.json"

    results = {}
    for i, instance in enumerate(subset):
        print(f"[Story-only] Processing {instance['id']} ({i+1}/{len(subset)})...")
        try:
            results[instance["id"]] = run_instance_story_only(instance)
        except Exception as e:
            results[instance["id"]] = {"error": str(e)}

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved story-only results for {len(subset)} instances to {output_path}")


if __name__ == "__main__":
    run_batch_story_only(n=500, mode="stratified")