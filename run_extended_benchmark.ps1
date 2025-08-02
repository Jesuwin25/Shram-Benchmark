<#
run_extended_benchmark.ps1
--------------------------
Runs the extended benchmark twice:
  • GPU run  → http://localhost:11434
  • CPU run  → http://localhost:11435  (start a CPU-only server:  `ollama serve --no-gpu --port 11435`)
Outputs JSON files extended_results_<host>_GPU.json and ..._CPU.json
#>

# ----- configurable section -----
# List of models to benchmark.  These names must match the smallest,
# up‑to‑date models installed in your local Ollama instance.  Run
# `ollama ls` to verify the available tags.  The list below targets
# Gemma 3 1B, Qwen 2.5 1.5B and Llama 3.2 1B – the most compact
# models currently offered on Ollama (as of mid‑2025) for each
# architecture.
$models        = @('gemma3:4b','qwen2.5:3b','llama3.2:3b')

# Sampling temperatures to test.  Using multiple temperatures allows you
# to evaluate factuality and diversity at different creativity levels.
$temperatures  = @('0.0','0.7','1.0')

# Number of tokens to generate for each prompt.  Adjust this if your
# benchmark requires longer or shorter outputs.
$numPredict    = 256

# Prompt to evaluate across all models and temperatures.  This prompt
# encourages the model to compose a brief narrative using the provided
# context.  The retrieval context will be prepended automatically by
# benchmark_extended.py, so instruct the model to use it.
$prompt        = "Write a brief story about Shram's mission and its impact on work using the provided context."

# Path to the factuality dataset used for scoring model outputs.  The
# dataset should reside in the same directory as this script or be
# specified with a relative/absolute path.  Use 'shram_ai_dataset.json'
# for the Shram.io / Shram.ai mission dataset.
$factsPath     = 'shram_ai_dataset.json'

# Weights for the composite score.  Factuality and efficiency are
# normalised and combined using these weights.  Adjust the numbers to
# prioritise factual accuracy or performance.
$weightFact    = 0.6
$weightEff     = 0.4
# --------------------------------

function Run-Bench {
    param($baseUrl, $suffix)

    $hostName = $env:COMPUTERNAME
    $outFile  = "extended_results_${hostName}_${suffix}.json"

    # Build argument array
    $args = @(
        '--base-url', $baseUrl,
        '--models') + $models +
        '--temperatures' + $temperatures + @(
        '--prompt', $prompt,
        '--num-predict', $numPredict,
        '--factuality-dataset', $factsPath,
        '--weight-factuality', $weightFact,
        '--weight-efficiency', $weightEff,
        '--output', $outFile
    )

    Write-Host "Running benchmark on $suffix using $baseUrl..."
    python .\benchmark_extended.py @args
    Write-Host "Benchmark complete.  Results saved to $outFile"
}

Run-Bench 'http://localhost:11434' 'GPU'

# Note: A separate CPU run is not configured here because Ollama does not
# expose explicit flags to force CPU-only execution.  If you wish to
# benchmark CPU performance, consider running Ollama on a machine without
# a compatible GPU or using an alternative inference engine.
