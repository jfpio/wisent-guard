#!/usr/bin/env python3
"""
Performance benchmarking script for Wisent Guard.

This script measures the actual performance overhead of Wisent Guard
after the debugging demo has identified and fixed any issues.

Usage:
    python benchmarking/performance_benchmark.py
"""

import time
import torch
import numpy as np
import pandas as pd
from statistics import mean, stdev
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmarking_demo import CONFIG, print_section, print_success, print_error


def load_components():
    """Load all necessary components for benchmarking."""
    print_section("LOADING COMPONENTS FOR BENCHMARKING")

    # Import here to avoid issues if debugging demo hasn't run
    from benchmarking_demo import test_2_model_loading, test_6_wisent_model_wrapper
    from wisent_guard.core import Classifier, Layer

    # Load model and tokenizer
    model, tokenizer = test_2_model_loading()
    if model is None:
        return None

    # Create Wisent model wrapper
    wrapper_result = test_6_wisent_model_wrapper(model, tokenizer)
    if wrapper_result is None:
        return None

    wisent_model, layer_obj, _ = wrapper_result

    # Create a simple classifier for testing (won't be trained)
    classifier = Classifier(
        model_type="logistic", device=CONFIG["device"], threshold=0.5
    )

    print_success("All components loaded successfully")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "wisent_model": wisent_model,
        "layer_obj": layer_obj,
        "classifier": classifier,
    }


def benchmark_baseline(model, tokenizer, prompts, num_trials=5):
    """Benchmark baseline model performance without Wisent Guard."""
    print_section("BASELINE BENCHMARK")

    results = []

    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}")
        trial_times = []

        for prompt in prompts:
            # Measure inference time
            start_time = time.time()

            inputs = tokenizer(prompt, return_tensors="pt").to(CONFIG["device"])

            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=CONFIG["max_new_tokens"],
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )

            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            trial_times.append(inference_time)

            # Clean up
            if CONFIG["device"] == "mps":
                torch.mps.empty_cache()
            elif CONFIG["device"] == "cuda":
                torch.cuda.empty_cache()

        results.append(
            {
                "trial": trial + 1,
                "mean_latency_ms": mean(trial_times),
                "std_latency_ms": stdev(trial_times) if len(trial_times) > 1 else 0,
                "total_time_ms": sum(trial_times),
            }
        )

    return results


def benchmark_with_activation_extraction(
    model, tokenizer, wisent_model, layer_obj, prompts, num_trials=5
):
    """Benchmark model performance with activation extraction (simulating Wisent Guard overhead)."""
    print_section("ACTIVATION EXTRACTION BENCHMARK")

    results = []

    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}")
        trial_times = []

        for prompt in prompts:
            # Measure inference time with activation extraction
            start_time = time.time()

            # Generate with activation extraction
            inputs = tokenizer(prompt, return_tensors="pt").to(CONFIG["device"])

            with torch.no_grad():
                # First get the generation
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=CONFIG["max_new_tokens"],
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )

                # Then extract activations (simulating guard overhead)
                activations = wisent_model.extract_activations(prompt, layer_obj)

                # Simulate classifier processing
                if activations is not None:
                    # Simple computation to simulate classifier
                    _ = torch.mean(activations).item()

            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            trial_times.append(inference_time)

            # Clean up
            if CONFIG["device"] == "mps":
                torch.mps.empty_cache()
            elif CONFIG["device"] == "cuda":
                torch.cuda.empty_cache()

        results.append(
            {
                "trial": trial + 1,
                "mean_latency_ms": mean(trial_times),
                "std_latency_ms": stdev(trial_times) if len(trial_times) > 1 else 0,
                "total_time_ms": sum(trial_times),
            }
        )

    return results


def analyze_performance(baseline_results, guard_results):
    """Analyze the performance difference between baseline and guard."""
    print_section("PERFORMANCE ANALYSIS")

    baseline_df = pd.DataFrame(baseline_results)
    guard_df = pd.DataFrame(guard_results)

    baseline_mean = baseline_df["mean_latency_ms"].mean()
    guard_mean = guard_df["mean_latency_ms"].mean()

    overhead_ms = guard_mean - baseline_mean
    overhead_percent = (overhead_ms / baseline_mean) * 100
    cost_multiplier = guard_mean / baseline_mean

    print(f"Baseline mean latency: {baseline_mean:.2f} ms")
    print(f"Guard mean latency: {guard_mean:.2f} ms")
    print(f"Overhead: {overhead_ms:.2f} ms ({overhead_percent:.1f}%)")
    print(f"Cost multiplier: {cost_multiplier:.2f}x")

    # Statistical significance test
    try:
        from scipy import stats

        baseline_latencies = baseline_df["mean_latency_ms"].values
        guard_latencies = guard_df["mean_latency_ms"].values

        t_stat, p_value = stats.ttest_ind(baseline_latencies, guard_latencies)
        print(f"Statistical significance (p-value): {p_value:.4f}")

        significant = p_value < 0.05
        print(f"Statistically significant: {'Yes' if significant else 'No'}")

    except ImportError:
        print("scipy not available, skipping statistical test")

    return {
        "baseline_mean": baseline_mean,
        "guard_mean": guard_mean,
        "overhead_ms": overhead_ms,
        "overhead_percent": overhead_percent,
        "cost_multiplier": cost_multiplier,
    }


def run_performance_benchmark():
    """Run the complete performance benchmark."""
    print_section("WISENT GUARD PERFORMANCE BENCHMARK")

    # Load components
    components = load_components()
    if components is None:
        print_error("Failed to load components")
        return

    # Test prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain machine learning briefly.",
        "How do you make coffee?",
        "What is the weather like?",
        "Tell me about space exploration.",
    ]

    num_trials = 3  # Reduced for faster local testing

    # Benchmark baseline
    baseline_results = benchmark_baseline(
        components["model"], components["tokenizer"], test_prompts, num_trials
    )

    # Benchmark with activation extraction
    guard_results = benchmark_with_activation_extraction(
        components["model"],
        components["tokenizer"],
        components["wisent_model"],
        components["layer_obj"],
        test_prompts,
        num_trials,
    )

    # Analyze results
    analysis = analyze_performance(baseline_results, guard_results)

    # Print final summary
    print_section("FINAL SUMMARY FOR CLIENT")
    print(
        f"🎯 Performance Impact: {analysis['overhead_percent']:.1f}% latency increase"
    )
    print(
        f"💰 Cost Multiplier: {analysis['cost_multiplier']:.2f}x baseline compute cost"
    )
    print(f"⏱️  Additional Latency: {analysis['overhead_ms']:.1f}ms per request")

    if analysis["overhead_percent"] < 20:
        print("✅ Low impact - suitable for production deployment")
    elif analysis["overhead_percent"] < 50:
        print("⚠️  Moderate impact - consider optimization")
    else:
        print("❌ High impact - optimization recommended")

    return analysis


if __name__ == "__main__":
    try:
        print("🚀 Starting performance benchmark...")
        print("📝 Note: Run benchmarking_demo.py first to ensure components work")

        analysis = run_performance_benchmark()

        if analysis:
            print("\n🎉 Performance benchmark completed successfully!")
        else:
            print("\n💥 Performance benchmark failed")

    except KeyboardInterrupt:
        print("\n\n⏹️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error in benchmark: {e}")
        import traceback

        traceback.print_exc()
