import time
import ollama
import psutil
import pandas as pd   # pip install pandas openpyxl

# ----------------------------
# 1. Define test questions + answers
# ----------------------------
test_cases = [
    {"q": "Who wrote the play Hamlet?", "a": "William Shakespeare"},
    {"q": "What is the capital of France?", "a": "Paris"},
    {"q": "What is 12 multiplied by 8?", "a": "96"},
    {"q": "In which year did the first man land on the moon?", "a": "1969"},
    {"q": "What is the boiling point of water in Celsius?", "a": "100"},
]

# ----------------------------
# 2. Run benchmark
# ----------------------------
def benchmark_model(model_name):
    print(f"\n🔎 Benchmarking model: {model_name}")
    results = []
    correct = 0
    total_latency = 0

    for case in test_cases:
        # Measure CPU + memory before
        cpu_before = psutil.cpu_percent(interval=None)
        mem_before = psutil.virtual_memory().used / (1024**3)

        start = time.time()
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": case["q"]}]
        )
        end = time.time()

        # Measure CPU + memory after
        cpu_after = psutil.cpu_percent(interval=None)
        mem_after = psutil.virtual_memory().used / (1024**3)

        latency = end - start
        total_latency += latency

        output = response["message"]["content"]
        # Simple accuracy check: is expected answer in output?
        is_correct = case["a"].lower() in output.lower()
        if is_correct:
            correct += 1

        results.append({
            "model": model_name,
            "question": case["q"],
            "expected": case["a"],
            "output": output.strip(),
            "correct": is_correct,
            "latency_sec": round(latency, 2),
            "cpu%_delta": cpu_after - cpu_before,
            "mem_used_GB_delta": round(mem_after - mem_before, 2)
        })

    accuracy = (correct / len(test_cases)) * 100
    avg_latency = total_latency / len(test_cases)

    print(f"\n📊 Summary for {model_name}:")
    print(f"   Accuracy: {accuracy:.1f}%")
    print(f"   Avg Latency: {avg_latency:.2f} sec\n")

    return results


# ----------------------------
# 3. Run on multiple models
# ----------------------------
models = ["llama3:latest", "llama2:latest"]   # change depending on what you have
all_results = []

for m in models:
    results = benchmark_model(m)
    all_results.extend(results)

# ----------------------------
# 4. Save to CSV and Excel
# ----------------------------
df = pd.DataFrame(all_results)
df.to_csv("ollama_benchmark_results.csv", index=False)
df.to_excel("ollama_benchmark_results.xlsx", index=False)

print("\n✅ Results saved as 'ollama_benchmark_results.csv' and 'ollama_benchmark_results.xlsx'")

