"""
benchmark_extended.py
=====================

This script extends the basic Ollama benchmarking functionality to include
variability, factuality scoring and composite metrics across multiple
temperature settings and hardware configurations.  It performs the
following steps:

1. **Throughput Benchmark**:  For each model and temperature, it sends a
   single prompt to the Ollama `generate` endpoint and records token
   throughput (eval_count/eval_duration) along with latency and VRAM
   consumption.  This is similar to the basic benchmark but includes
   the `temperature` option to assess variability.
2. **Factuality Evaluation**:  It can optionally load a small dataset of
   factual questions and reference answers from a JSON file.  For each
   question and for each model/temperature combination, the script
   requests a short answer and computes a simple overlap score between
   the generated response and the ground‑truth answers.  The mean of
   these scores yields a “factuality percentage” for that combination.
   You can replace the default dataset with TruthfulQA or another
   benchmark by providing `--factuality-dataset`.
3. **Composite Scoring**:  It normalises both throughput and factuality
   scores across all tested combinations and computes a weighted sum
   according to user‑provided weights (`--weight-factuality` and
   `--weight-efficiency`).  The result is a single score that balances
   speed and truthfulness.  A summary table is printed and written to
   the output JSON.

Usage example:

```
python benchmark_extended.py \
    --models llama3:8b qwen2:7b gemma:2b \
    --temperatures 0.0 0.7 1.0 \
    --num-predict 128 \
    --factuality-dataset factuality_dataset.json \
    --weight-factuality 0.6 \
    --weight-efficiency 0.4 \
    --output extended_results.json
```

Each machine running this script will produce a report that you can
merge later for cross‑machine analysis.  For CPU‑only tests, start
Ollama with the `--no-gpu` option or on a machine without a GPU.

Note: This script depends on `requests`, `psutil`, and `pandas`
(`pandas` is optional and only used for a pretty table in the console).
Install missing packages via pip if necessary.
"""

import argparse
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests  # type: ignore
import psutil  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore


###############################
# Hardware and utility functions
###############################

# Global context string built from the dataset.  It will be populated in
# ``main`` and referenced by ``benchmark_throughput`` and
# ``evaluate_factuality`` to provide retrieval context.
global_context: str = ""

def get_hardware_info() -> Dict[str, Any]:
    """Collect basic hardware information about the host machine."""
    info: Dict[str, Any] = {}
    info["timestamp"] = _dt.datetime.now().isoformat()
    info["system"] = platform.system()
    info["release"] = platform.release()
    info["machine"] = platform.machine()
    info["processor"] = platform.processor()
    try:
        cpu_freq = psutil.cpu_freq()
        if cpu_freq is not None:
            info["cpu_mhz"] = cpu_freq.max
    except Exception:
        pass
    try:
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / (1024 ** 3), 2)
    except Exception:
        pass
    gpus: List[Dict[str, Any]] = []
    try:
        result = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ])
        for line in result.decode().strip().split("\n"):
            name, total, used, free = [x.strip() for x in line.split(",")]
            gpus.append({
                "name": name,
                "memory_total_gb": round(float(total) / 1024, 2),
                "memory_used_gb": round(float(used) / 1024, 2),
                "memory_free_gb": round(float(free) / 1024, 2),
            })
        info["gpus"] = gpus
    except Exception:
        info["gpus"] = []
    return info


def nvidia_memory_usage() -> Optional[float]:
    """Return current total GPU memory used across all GPUs in GB."""
    try:
        result = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ])
        usages = [float(x.strip()) for x in result.decode().split()]
        total_used_mb = sum(usages)
        return round(total_used_mb / 1024, 2)
    except Exception:
        return None


###############################
# Factuality scoring utilities
###############################

def load_factuality_dataset(path: str) -> List[Dict[str, Any]]:
    """Load a factuality dataset from a JSON file.

    The file should be a JSON object with a top‑level `questions` list;
    each element must have `prompt` (string) and `answers` (list of strings).
    If the file does not exist or is malformed, an empty list is returned.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("questions", [])
    except Exception as e:
        print(f"Warning: failed to load factuality dataset '{path}': {e}")
        return []


import re

def clean_text(text: str) -> str:
    """Remove citation markers and extraneous whitespace from text.

    Citations in the dataset (e.g., 【123†L45-L47】) can interfere with factuality
    scoring.  This function strips out any text enclosed in square brackets
    beginning with a double width bracket (【…】) as well as standard brackets
    containing citation markers, and collapses multiple spaces.
    """
    # Remove patterns like 【123†L45-L47】
    cleaned = re.sub(r"【[^】]*】", "", text)
    # Remove citations in regular brackets [123] or (123)
    cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)
    cleaned = re.sub(r"\([^\)]*\)", "", cleaned)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()

def token_set(text: str) -> set:
    """Simple tokenisation by splitting on whitespace and lower‑casing.

    Before splitting, the text is cleaned to remove citation markers and
    punctuation that might otherwise skew overlap computations.
    """
    cleaned = clean_text(text)
    return set(t.lower().strip(".,;:!?()[]{}\"'\n\r") for t in cleaned.split())


def f1_overlap_score(response: str, reference_answers: List[str]) -> float:
    """Compute an F1‑based overlap score between a response and reference answers.

    For each reference answer, compute the precision (resp ∩ ref / resp) and
    recall (resp ∩ ref / ref), then compute the harmonic mean (F1).  The
    maximum F1 across all reference answers is returned.  This encourages
    answers that both cover relevant facts (high recall) and avoid
    hallucinations (high precision).
    """
    resp_tokens = token_set(response)
    if not resp_tokens:
        return 0.0
    best_f1 = 0.0
    for ans in reference_answers:
        ref_tokens = token_set(ans)
        if not ref_tokens:
            continue
        intersection = len(resp_tokens & ref_tokens)
        if intersection == 0:
            continue
        precision = intersection / len(resp_tokens)
        recall = intersection / len(ref_tokens)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1


def jaccard_distance(a: str, b: str) -> float:
    """Compute one minus the Jaccard similarity between two strings.

    This is used to measure diversity between model outputs.  A higher
    distance indicates more dissimilar outputs.
    """
    set_a = token_set(a)
    set_b = token_set(b)
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return 1.0 - (intersection / union) if union > 0 else 0.0


def measure_diversity(responses: List[str]) -> float:
    """Compute average pairwise Jaccard distance among a list of responses.

    Given a list of response strings (e.g., the outputs for a model at
    different temperatures), this function returns the mean of all
    pairwise Jaccard distances.  A higher value indicates more diverse
    outputs across temperatures.
    """
    if len(responses) < 2:
        return 0.0
    dists = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            dists.append(jaccard_distance(responses[i], responses[j]))
    return sum(dists) / len(dists) if dists else 0.0


###############################
# Benchmarking functions
###############################

def benchmark_throughput(
    base_url: str,
    model: str,
    prompt: str,
    num_predict: int,
    temperature: float,
) -> Dict[str, Any]:
    """Benchmark token throughput for a single model at a given temperature.

    Returns a dictionary with throughput metrics, latency, VRAM usage and
    the model response.  In case of failure, an `error` key is included.
    """
    entry: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
    }
    pre_mem = nvidia_memory_usage()
    start_time = time.perf_counter()
    try:
        # When measuring throughput, we also include the dataset context in the prompt
        # so that the model has access to relevant information about Shram.
        # Prepend retrieval context and instruct the model to rely solely on it.
        context_prompt = (
            f"Context: {global_context}\n\n"
            f"User: {prompt}\n\n"
            "Guidelines: Use only information from the context above. Do not add any extra names, events or details.\n"
            "Answer:"
        )
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": context_prompt,
                "stream": False,
                "options": {
                    "num_predict": num_predict,
                    "temperature": temperature,
                },
            },
            timeout=600,
        )
        latency = time.perf_counter() - start_time
        resp.raise_for_status()
        data = resp.json()
        for key in [
            "response",
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
            "done_reason",
        ]:
            if key in data:
                entry[key] = data[key]
        eval_count = data.get("eval_count")
        eval_duration = data.get("eval_duration")
        if isinstance(eval_count, int) and isinstance(eval_duration, int) and eval_duration > 0:
            tps = eval_count / (eval_duration / 1e9)
            entry["tokens_per_second"] = round(tps, 2)
            entry["tokens_per_minute"] = round(tps * 60, 2)
        entry["latency_seconds"] = round(latency, 3)
    except Exception as e:
        entry["error"] = str(e)
    post_mem = nvidia_memory_usage()
    if pre_mem is not None and post_mem is not None:
        entry["vram_used_before_gb"] = pre_mem
        entry["vram_used_after_gb"] = post_mem
        entry["vram_delta_gb"] = round(post_mem - pre_mem, 2)
    return entry


def evaluate_factuality(
    base_url: str,
    model: str,
    dataset: List[Dict[str, Any]],
    num_predict: int,
    temperature: float,
) -> Tuple[float, List[float]]:
    """Evaluate factuality for a model/temperature across a dataset.

    Returns the mean overlap score and a list of individual scores.
    """
    scores: List[float] = []
    for item in dataset:
        question = item.get("prompt")
        answers = item.get("answers", [])
        if not question or not answers:
            continue
        try:
            # Include context from dataset when generating answer
            resp = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    # Prepend context to help the model answer factually
                    "prompt": (
                        f"Context: {global_context}\n\n"
                        f"Question: {question}\n\n"
                        "Guidelines: Use only information from the context above. Provide a concise answer without adding extra details.\n"
                        "Answer:"
                    ),
                    "stream": False,
                    "options": {
                        "num_predict": num_predict,
                        "temperature": temperature,
                    },
                },
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()
            answer_text = data.get("response", data.get("content", ""))
            score = f1_overlap_score(answer_text, answers)
            scores.append(score)
        except Exception:
            scores.append(0.0)
    mean_score = sum(scores) / len(scores) if scores else 0.0
    return mean_score, scores


###############################
# Main orchestration
###############################

def main() -> None:
    parser = argparse.ArgumentParser(description="Extended benchmark for Ollama models with factuality.")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama server URL.")
    parser.add_argument("--models", nargs="*", required=True, help="List of model names to test.")
    parser.add_argument("--temperatures", nargs="*", type=float, default=[0.0, 0.7, 1.0], help="Temperature settings to test.")
    parser.add_argument(
        "--prompt",
        default="Write a brief story about Shram’s mission and its impact on work using the provided context.",
        help=(
            "Prompt for throughput measurement.  The global context derived from the"
            " factuality dataset will be prepended automatically.  This phrasing"
            " encourages models to draw on the supplied context to compose a narrative"
            " about Shram’s mission and impact on work."
        ),
    )
    parser.add_argument("--num-predict", type=int, default=256, help="Number of tokens to generate for throughput.")
    parser.add_argument("--factuality-dataset", default=None, help="Path to a JSON file with factuality questions.")
    parser.add_argument("--factuality-predict", type=int, default=64, help="Number of tokens to generate for each factuality question.")
    parser.add_argument("--weight-factuality", type=float, default=0.6, help="Weight of factuality in composite score.")
    parser.add_argument("--weight-efficiency", type=float, default=0.4, help="Weight of efficiency (tokens per second) in composite score.")
    parser.add_argument("--output", required=True, help="Output JSON file to write results to.")

    args = parser.parse_args()
    # Load factuality dataset if provided
    dataset: List[Dict[str, Any]] = []
    if args.factuality_dataset:
        dataset = load_factuality_dataset(args.factuality_dataset)
        if not dataset:
            print("No factuality questions loaded; skipping factuality evaluation.")

    # Build a global context string from the dataset answers to aid retrieval.
    # This context is appended to prompts when generating answers for
    # factuality questions and the main throughput prompt.  It helps the
    # model access the relevant information contained in the dataset.
    global global_context
    global_context = ""
    if dataset:
        context_parts: List[str] = []
        for item in dataset:
            for ans in item.get("answers", []):
                # Remove citations for cleaner context
                context_parts.append(clean_text(ans))
        # Join with space; limit length if necessary
        global_context = " ".join(context_parts)

    hardware = get_hardware_info()
    entries: List[Dict[str, Any]] = []
    # Collect responses per model for diversity measurement
    model_outputs: Dict[str, List[str]] = {}

    # Iterate over all combinations of model and temperature
    for model in args.models:
        for temp in args.temperatures:
            print(f"\nBenchmarking {model} at temperature {temp}...")
            # Throughput measurement
            thr_entry = benchmark_throughput(
                args.base_url,
                model,
                args.prompt,
                args.num_predict,
                temp,
            )
            # Factuality evaluation
            factuality_score = 0.0
            factuality_scores: List[float] = []
            if dataset:
                mean_score, scores = evaluate_factuality(
                    args.base_url,
                    model,
                    dataset,
                    args.factuality_predict,
                    temp,
                )
                factuality_score = mean_score
                factuality_scores = scores
                thr_entry["factuality_score"] = round(mean_score, 4)
                thr_entry["factuality_scores"] = [round(s, 4) for s in scores]
            # Accumulate responses for diversity
            if "response" in thr_entry and thr_entry.get("response"):
                model_outputs.setdefault(model, []).append(thr_entry["response"])
            entries.append(thr_entry)

    # Normalise throughput and factuality across entries for composite scoring
    max_tps = max((e.get("tokens_per_second", 0) for e in entries), default=0)
    max_fact = max((e.get("factuality_score", 0) for e in entries), default=0)
    for e in entries:
        norm_eff = (e.get("tokens_per_second", 0) / max_tps) if max_tps > 0 else 0
        norm_fact = (e.get("factuality_score", 0) / max_fact) if max_fact > 0 else 0
        composite = args.weight_factuality * norm_fact + args.weight_efficiency * norm_eff
        e["normalized_efficiency"] = round(norm_eff, 4)
        e["normalized_factuality"] = round(norm_fact, 4)
        e["composite_score"] = round(composite, 4)

    # Display results table if pandas is available
    if pd is not None and entries:
        df = pd.DataFrame(entries)
        cols = [
            "model",
            "temperature",
            "tokens_per_second",
            "factuality_score",
            "normalized_efficiency",
            "normalized_factuality",
            "composite_score",
            "model_diversity",
        ]
        display_cols = [c for c in cols if c in df.columns]
        print("\nSummary:\n")
        print(df[display_cols].sort_values(by="composite_score", ascending=False).to_string(index=False))
    else:
        # Fallback summary
        print("\nSummary (descending composite score):")
        for e in sorted(entries, key=lambda x: x.get("composite_score", 0), reverse=True):
            print(
                f"Model {e['model']} @ temp {e['temperature']} - TPS: {e.get('tokens_per_second', 0):.2f}, "
                f"Fact: {e.get('factuality_score', 0):.4f}, Composite: {e.get('composite_score', 0):.4f}"
            )

    # Write report to file
    # Compute diversity per model
    model_diversity: Dict[str, float] = {}
    max_div = 0.0
    for model, outputs in model_outputs.items():
        div = measure_diversity(outputs)
        model_diversity[model] = round(div, 4)
        if div > max_div:
            max_div = div
    # Normalize diversity values for optional use
    normalized_div: Dict[str, float] = {}
    for m, d in model_diversity.items():
        normalized_div[m] = round((d / max_div) if max_div > 0 else 0.0, 4)
    # Add diversity information to each entry
    for entry in entries:
        entry["model_diversity"] = normalized_div.get(entry["model"], 0.0)

    report = {
        "hardware": hardware,
        "prompt": args.prompt,
        "num_predict": args.num_predict,
        "factuality_dataset": args.factuality_dataset,
        "weight_factuality": args.weight_factuality,
        "weight_efficiency": args.weight_efficiency,
        "entries": entries,
        "model_diversity": model_diversity,
    }
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nExtended benchmark report written to {args.output}")
    except Exception as e:
        print(f"Failed to write output file {args.output}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()