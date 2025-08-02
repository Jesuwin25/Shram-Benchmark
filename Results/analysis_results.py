"""
analysis_results.py
====================

This script aggregates and analyses benchmark results produced by
`benchmark_extended.py` across multiple machines.  It recursively
searches a given directory for JSON files whose names match
`extended_results_*.json` (both GPU and CPU reports are supported),
loads the entries from each file, and computes summary statistics.

Key features:

* **Recursive discovery** – It walks the provided root folder and
  collects all JSON files matching the `extended_results_` prefix.
* **DataFrame construction** – If pandas is available, it builds a
  DataFrame for convenient grouping and summarisation.  A fallback to
  basic Python lists ensures the script still runs without pandas.
* **Per‑model and per‑temperature analysis** – For each unique
  combination of model and temperature, it reports the mean and
  standard deviation of throughput (`tokens_per_second`), factuality
  scores and composite scores across all systems.
* **Overall rankings** – It computes the average composite score for
  each model (across temperatures) and lists the top performers.
* **VRAM and latency statistics** – It summarises VRAM usage (delta
  memory) and latency across the collected entries.

Usage example:

```bash
python analysis_results.py --root "D:/New folder (2)/Shram Benchmark/Results"
```

The script prints results to the console.  You can redirect output to
a file for inclusion in your report.  Feel free to modify or extend
the analysis as needed for your specific reporting requirements.
"""

import argparse
import json
import os
import statistics
from typing import Dict, List, Tuple

try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None  # type: ignore


def collect_json_files(root: str) -> List[str]:
    """Recursively find all JSON files starting with 'extended_results_' in root."""
    matches: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.startswith("extended_results_") and fname.endswith(".json"):
                matches.append(os.path.join(dirpath, fname))
    return matches


def load_entries(path: str) -> List[Dict]:
    """Load the 'entries' list from a single benchmark JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("entries", [])
        # Include system identifier based on file name or hardware info
        host_id = os.path.splitext(os.path.basename(path))[0].replace("extended_results_", "")
        for entry in entries:
            entry["source_file"] = os.path.basename(path)
            entry["host_id"] = host_id
        return entries
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return []


def summarise(entries: List[Dict]) -> None:
    """Print summary statistics from a list of benchmark entries."""
    if not entries:
        print("No entries to summarise.")
        return

    # Keys to consider for numeric metrics
    numeric_keys = [
        "tokens_per_second",
        "factuality_score",
        "composite_score",
        "normalized_factuality",
        "normalized_efficiency",
        "model_diversity",
        "vram_delta_gb",
        "latency_seconds",
    ]

    # Convert to DataFrame if pandas is available
    if pd is not None:
        df = pd.DataFrame(entries)
        # Convert to numeric where possible
        for key in numeric_keys:
            if key in df.columns:
                df[key] = pd.to_numeric(df[key], errors="coerce")

        print(f"Collected {len(df)} entries from {df['host_id'].nunique()} systems.")

        # Group by model and temperature
        group_cols = ["model", "temperature"]
        grouped = df.groupby(group_cols)[numeric_keys].agg(['mean', 'std']).round(4)
        print("\nPer‑model and temperature summary (mean ± std):\n")
        print(grouped)

        # Overall rankings by model (averaging across temperatures)
        print("\nAverage composite score by model (across temperatures):\n")
        comp_by_model = df.groupby("model")["composite_score"].mean().sort_values(ascending=False)
        print(comp_by_model)

        # VRAM and latency overview
        if "vram_delta_gb" in df.columns:
            print("\nVRAM usage statistics (GB):")
            print(df["vram_delta_gb"].describe().round(4))
        if "latency_seconds" in df.columns:
            print("\nLatency statistics (seconds):")
            print(df["latency_seconds"].describe().round(4))

    else:
        # Fallback: basic summary without pandas
        print(f"Collected {len(entries)} entries.")
        # Build nested dict for model/temp combinations
        stats: Dict[Tuple[str, float], Dict[str, List[float]]] = {}
        for e in entries:
            key = (str(e.get("model")), float(e.get("temperature", 0.0)))
            if key not in stats:
                stats[key] = {k: [] for k in numeric_keys}
            for k in numeric_keys:
                val = e.get(k)
                if isinstance(val, (int, float)):
                    stats[key][k].append(val)

        # Print summary
        for (model, temp), metrics in stats.items():
            print(f"\nModel {model} @ temp {temp}:")
            for k, values in metrics.items():
                if values:
                    mean = statistics.mean(values)
                    std = statistics.pstdev(values) if len(values) > 1 else 0.0
                    print(f"  {k}: {mean:.4f} ± {std:.4f}")

        # Overall composite ranking
        composite_dict: Dict[str, List[float]] = {}
        for e in entries:
            model = str(e.get("model"))
            comp = e.get("composite_score")
            if isinstance(comp, (int, float)):
                composite_dict.setdefault(model, []).append(comp)
        print("\nAverage composite score by model:")
        for model, comps in composite_dict.items():
            avg = statistics.mean(comps)
            print(f"  {model}: {avg:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate and analyse extended benchmark results across machines.")
    parser.add_argument("--root", required=True, help="Root directory containing extended_results JSON files.")
    args = parser.parse_args()

    json_files = collect_json_files(args.root)
    if not json_files:
        print(f"No extended_results JSON files found under {args.root}.")
        return

    all_entries: List[Dict] = []
    for path in json_files:
        entries = load_entries(path)
        all_entries.extend(entries)

    summarise(all_entries)


if __name__ == "__main__":
    main()