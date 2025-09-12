# functions to run encoding over all ASP instances and retrieve facts entailed by instance

import os
import json
from subprocess import run, PIPE, TimeoutExpired

def call_clingo(clingo, input_names, timeout):
    cmd = [clingo, "--warn=none", "--outf=2"] + input_names
    output = run(cmd, stdout=PIPE, stderr=PIPE, timeout=timeout)
    if output.stderr:
        raise RuntimeError(f"Clingo error: {output.stderr.decode()}")
    return json.loads(output.stdout)

def run_instances(clingo_bin, encoding, instance_dirs, timeout=30, out_json="results.json"):
    results = {}

    for inst_dir in instance_dirs:
        for fname in sorted(os.listdir(inst_dir)):
            if not fname.endswith(".lp"):
                continue
            fpath = os.path.join(inst_dir, fname)
            try:
                data = call_clingo(clingo_bin, [encoding, fpath], timeout)
                instance_result = {}

                if data["Result"] == "UNSATISFIABLE":
                    instance_result["UNSAT"] = True
                else:
                    atoms = []
                    for call in data["Call"]:
                        for witness in call.get("Witnesses", []):
                            atoms.extend(witness.get("Value", []))

                    # group by predicate name
                    preds = {}
                    for atom in atoms:
                        pred = atom.split("(")[0] if "(" in atom else atom
                        preds.setdefault(pred, []).append(atom)
                    instance_result = preds

                results[fname] = instance_result

            except TimeoutExpired:
                results[fname] = {"TIMEOUT": True}
            except Exception as e:
                results[fname] = {"ERROR": str(e)}

    # write to JSON file
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results written to {out_json}")

if __name__ == "__main__":
    
    encoding = "src/tg_reasoner.lp"
    instance_dirs = ["ASPinstances/TGQA", "ASPinstances/TimeQA"]
    run_instances("clingo", encoding, instance_dirs, timeout=1000, out_json="results/asp_results.json")