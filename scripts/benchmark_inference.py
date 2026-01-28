import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from tqdm import tqdm

from gemmaqa.inference.model import generate_response, load_model_for_inference


def run_benchmark(
    checkpoint_path: str,
    base_model: str,
    data_path: str = "data/test_subset.json",
    num_samples: int = 1000,
    max_new_tokens: int = 50,
):
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    print(f"Loading data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if len(dataset) > num_samples:
        dataset = dataset[:num_samples]

    print(f"Selected {len(dataset)} samples for benchmarking.")

    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model_for_inference(
        checkpoint_path=checkpoint_path, base_model_name=base_model
    )

    # warmup
    print("Performing warm-up (5 queries)...")
    for i in range(5):
        _ = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt="Context: Test context.\n\nQuestion: Warmup?",
            max_new_tokens=10,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # variables for collecting data
    latencies = []
    token_counts = []

    print(f"\nStarting benchmark on {len(dataset)} items...")

    for item in tqdm(dataset, desc="Benchmarking"):
        context = item.get("context", "")
        question = item.get("question", "")

        prompt = f"Context: {context}\n\nQuestion: {question}"

        start_time = time.perf_counter()

        response_text = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            temperature=0.000001,
            max_new_tokens=max_new_tokens,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Count tokens in response
        generated_tokens = tokenizer.encode(response_text, add_special_tokens=False)
        num_tokens = len(generated_tokens)

        if num_tokens == 0:
            num_tokens = 1

        latencies.append(duration)
        token_counts.append(num_tokens)

    # statictics
    total_time = sum(latencies)
    total_tokens = sum(token_counts)

    avg_latency = statistics.mean(latencies)

    avg_tokens = statistics.mean(token_counts)

    # Throughput = total number of tokens / total time
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0

    # Queries per second
    queries_per_second = len(dataset) / total_time if total_time > 0 else 0

    # Time per token (in miliseconds)
    ms_per_token = (total_time * 1000) / total_tokens if total_tokens > 0 else 0

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS (With Token Metrics)")
    print("=" * 60)
    print(f"Model:          {checkpoint_path if checkpoint_path else base_model}")
    print(f"Samples:        {len(latencies)}")
    print(f"Total Time:     {total_time:.2f} s")
    print(f"Total Tokens:   {total_tokens}")
    print("-" * 30)
    print(f"Avg Response Length:   {avg_tokens:.2f} tokens")
    print(f"Avg Latency (Time/Q):  {avg_latency:.4f} s")
    print("-" * 30)
    print(f"Throughput (Speed):    {tokens_per_second:.2f} tokens/s")
    print(f"Time per Token:        {ms_per_token:.2f} ms/token")
    print(f"Queries per Second:    {queries_per_second:.2f} queries/s")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark inference speed with token counting."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint/adapter",
    )
    parser.add_argument(
        "--base-model", type=str, default="google/gemma-3-1b-it", help="Base model name"
    )
    parser.add_argument(
        "--data", type=str, default="data/test_subset.json", help="Path to test dataset"
    )
    parser.add_argument(
        "--num-samples", type=int, default=1000, help="Number of samples to run"
    )

    args = parser.parse_args()

    run_benchmark(
        checkpoint_path=args.checkpoint,
        base_model=args.base_model,
        data_path=args.data,
        num_samples=args.num_samples,
    )
