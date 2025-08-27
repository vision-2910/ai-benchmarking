import json
import subprocess
import time
import difflib

# Config
BENCHMARK_FILE = "zero_shot_benchmark.json"
MODELS = ["llama2:latest", "llama3:latest"]

# Simple fuzzy match for evaluation
def is_correct(output, expected):
    output = output.strip().lower()
    expected = expected.strip().lower()

    if expected in output:
        return True
    # Allow near matches for summarization/reasoning
    ratio = difflib.SequenceMatcher(None, output, expected).ratio()
    return ratio > 0.6

def run_prompt(model, prompt):
    start = time.time()
    try:
        result = subprocess.run(
            #["ollama", "run", model, prompt],
            ["docker", "exec", "-i", "ollama", "ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=120
        )
    except subprocess.TimeoutExpired:
        return None, None

    end = time.time()
    latency = end - start
    return result.stdout.strip(), latency

def benchmark_model(model, tasks):
    print(f"\n🔎 Benchmarking model: {model}")
    total, correct, latencies = 0, 0, []

    for task in tasks:
        output, latency = run_prompt(model, task["prompt"])
        if output is None:
            print(f"  ⚠️ Timeout on {task['id']}")
            continue

        total += 1
        latencies.append(latency)
        success = is_correct(output, task["expected"])
        if success:
            correct += 1

        print(f"\nTask: {task['id']}")
        print(f"Prompt: {task['prompt']}")
        print(f"Expected: {task['expected']}")
        print(f"Output: {output}")
        print(f"Latency: {latency:.2f}s | {'✅ Correct' if success else '❌ Incorrect'}")

    accuracy = (correct / total) * 100 if total > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    print(f"\n📊 Summary for {model}:")
    print(f"   Accuracy: {accuracy:.1f}%")
    print(f"   Avg Latency: {avg_latency:.2f} sec")

def main():
    with open(BENCHMARK_FILE, "r") as f:
        tasks = json.load(f)

    for model in MODELS:
        benchmark_model(model, tasks)

if __name__ == "__main__":
    main()

