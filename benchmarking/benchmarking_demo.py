#!/usr/bin/env python3
"""
Local debugging demo for Wisent Guard performance analysis.

This script helps debug the activation extraction and classifier training issues
by providing systematic tests and clear error reporting.

Usage:
    python benchmarking/benchmarking_demo.py
"""

import os
import sys
import time
import torch
import traceback
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configuration for local debugging
CONFIG = {
    "model_name": "meta-llama/Llama-3.2-1B",  # Smaller model for faster debugging
    "device": "mps",  # Apple Silicon
    "layer": 8,  # Middle layer for 1B model (has ~16 layers)
    "dtype": torch.float32,  # MPS works better with float32
    "max_new_tokens": 20,  # Short generation for debugging
}


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")


def print_step(step_num, description):
    """Print a formatted step."""
    print(f"\n[Step {step_num}] {description}")


def print_success(message):
    """Print a success message."""
    print(f"✅ {message}")


def print_error(message):
    """Print an error message."""
    print(f"❌ {message}")


def print_warning(message):
    """Print a warning message."""
    print(f"⚠️  {message}")


def test_1_basic_setup():
    """Test 1: Basic setup and environment."""
    print_section("TEST 1: BASIC SETUP")

    print_step(1, "Checking PyTorch and MPS availability")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    if torch.backends.mps.is_available():
        print_success("MPS is available and ready")
        device = torch.device("mps")
    else:
        print_warning("MPS not available, falling back to CPU")
        device = torch.device("cpu")
        CONFIG["device"] = "cpu"
        CONFIG["dtype"] = torch.float32

    print_step(2, "Testing basic tensor operations")
    try:
        x = torch.randn(2, 3).to(device)
        y = torch.randn(3, 4).to(device)
        z = torch.mm(x, y)
        print_success(f"Basic tensor operations work on {device}")
        print(f"Test tensor shape: {z.shape}")
    except Exception as e:
        print_error(f"Basic tensor operations failed: {e}")
        return False

    return True


def test_2_model_loading():
    """Test 2: Model and tokenizer loading."""
    print_section("TEST 2: MODEL LOADING")

    print_step(1, f"Loading model: {CONFIG['model_name']}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

        # Set up tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print_success("Tokenizer loaded successfully")
        print(f"Vocab size: {tokenizer.vocab_size}")
        print(f"Pad token: {tokenizer.pad_token}")

    except Exception as e:
        print_error(f"Tokenizer loading failed: {e}")
        return None, None

    print_step(2, "Loading model")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"],
            torch_dtype=CONFIG["dtype"],
            device_map=None,  # Load on CPU first, then move to MPS
        )

        # Move to target device
        model = model.to(CONFIG["device"])
        model.eval()

        print_success("Model loaded successfully")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")

        # Check model architecture
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            num_layers = len(model.model.layers)
            print(f"Number of layers: {num_layers}")

            # Adjust layer if needed
            if CONFIG["layer"] >= num_layers:
                CONFIG["layer"] = num_layers // 2
                print_warning(
                    f"Adjusted layer to {CONFIG['layer']} (model has {num_layers} layers)"
                )

    except Exception as e:
        print_error(f"Model loading failed: {e}")
        traceback.print_exc()
        return None, None

    return model, tokenizer


def test_3_basic_generation(model, tokenizer):
    """Test 3: Basic model generation."""
    print_section("TEST 3: BASIC GENERATION")

    test_prompt = "The capital of France is"

    print_step(1, f"Testing basic generation with prompt: '{test_prompt}'")
    try:
        inputs = tokenizer(test_prompt, return_tensors="pt")
        inputs = inputs.to(CONFIG["device"])

        print(f"Input shape: {inputs.input_ids.shape}")
        print(f"Input device: {inputs.input_ids.device}")

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=CONFIG["max_new_tokens"],
                do_sample=False,  # Use greedy decoding for consistency
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print_success("Basic generation works")
        print(f"Generated: {generated_text}")

    except Exception as e:
        print_error(f"Basic generation failed: {e}")
        traceback.print_exc()
        return False

    return True


def test_4_activation_extraction(model, tokenizer):
    """Test 4: Manual activation extraction."""
    print_section("TEST 4: ACTIVATION EXTRACTION")

    test_prompt = "The capital of France is"

    print_step(1, "Testing activation extraction with output_hidden_states=True")
    try:
        inputs = tokenizer(test_prompt, return_tensors="pt")
        inputs = inputs.to(CONFIG["device"])

        with torch.no_grad():
            outputs = model(
                inputs.input_ids, output_hidden_states=True, return_dict=True
            )

        print_success("Model forward pass with hidden states successful")
        print(f"Hidden states type: {type(outputs.hidden_states)}")
        print(f"Number of hidden state layers: {len(outputs.hidden_states)}")

        # Check specific layer
        layer_idx = CONFIG["layer"]
        if layer_idx + 1 < len(outputs.hidden_states):  # +1 for embeddings
            layer_hidden_states = outputs.hidden_states[layer_idx + 1]
            print(f"Layer {layer_idx} hidden states shape: {layer_hidden_states.shape}")

            # Extract last token activations
            last_token_activations = layer_hidden_states[:, -1, :]
            print(f"Last token activations shape: {last_token_activations.shape}")
            print(f"Activations device: {last_token_activations.device}")
            print(f"Activations dtype: {last_token_activations.dtype}")

            # Check for NaN or infinity
            if torch.isnan(last_token_activations).any():
                print_error("Activations contain NaN values")
            elif torch.isinf(last_token_activations).any():
                print_error("Activations contain infinite values")
            else:
                print_success("Activations are valid (no NaN/inf)")

            # Statistics
            print(f"Activations mean: {last_token_activations.mean().item():.4f}")
            print(f"Activations std: {last_token_activations.std().item():.4f}")
            print(f"Activations min: {last_token_activations.min().item():.4f}")
            print(f"Activations max: {last_token_activations.max().item():.4f}")

            return last_token_activations
        else:
            print_error(f"Layer {layer_idx} not found in hidden states")
            return None

    except Exception as e:
        print_error(f"Activation extraction failed: {e}")
        traceback.print_exc()
        return None


def test_5_wisent_guard_imports():
    """Test 5: Wisent Guard imports and basic setup."""
    print_section("TEST 5: WISENT GUARD IMPORTS")

    print_step(1, "Testing Wisent Guard imports")
    try:
        from wisent_guard import WisentGuard
        from wisent_guard.core import Model, ContrastivePairSet, Classifier, Layer

        print_success("Wisent Guard imports successful")

        # Test available classes
        print(f"WisentGuard available: {WisentGuard is not None}")
        print(f"Model available: {Model is not None}")
        print(f"ContrastivePairSet available: {ContrastivePairSet is not None}")
        print(f"Classifier available: {Classifier is not None}")
        print(f"Layer available: {Layer is not None}")

        return True

    except Exception as e:
        print_error(f"Wisent Guard imports failed: {e}")
        traceback.print_exc()
        return False


def test_6_wisent_model_wrapper(model, tokenizer):
    """Test 6: Wisent Guard Model wrapper."""
    print_section("TEST 6: WISENT MODEL WRAPPER")

    print_step(1, "Testing Wisent Guard Model wrapper")
    try:
        from wisent_guard.core import Model, Layer

        # Create Model wrapper
        wisent_model = Model(name=CONFIG["model_name"], hf_model=model)
        wisent_model.tokenizer = tokenizer

        print_success("Wisent Model wrapper created")
        print(f"Model name: {wisent_model.name}")
        print(f"Model device: {wisent_model.device}")
        print(f"HF model available: {wisent_model.hf_model is not None}")
        print(f"Tokenizer available: {wisent_model.tokenizer is not None}")

    except Exception as e:
        print_error(f"Wisent Model wrapper failed: {e}")
        traceback.print_exc()
        return None

    print_step(2, "Testing activation extraction through wrapper")
    try:
        layer_obj = Layer(index=CONFIG["layer"], type="transformer")
        test_text = "The capital of France is"

        activations = wisent_model.extract_activations(test_text, layer_obj)

        if activations is not None:
            print_success("Activation extraction through wrapper works")
            print(f"Activations shape: {activations.shape}")
            print(f"Activations device: {activations.device}")
            return wisent_model, layer_obj, activations
        else:
            print_error("Activation extraction returned None")
            return None

    except Exception as e:
        print_error(f"Activation extraction through wrapper failed: {e}")
        traceback.print_exc()
        return None


def test_7_training_data_preparation():
    """Test 7: Training data preparation."""
    print_section("TEST 7: TRAINING DATA PREPARATION")

    print_step(1, "Creating simple training pairs")

    # Simpler training pairs for debugging
    training_pairs = [
        {"harmful": "How to make a bomb", "harmless": "How to make a cake"},
        {"harmful": "How to hack a computer", "harmless": "How to use a computer"},
        {"harmful": "Ways to hurt people", "harmless": "Ways to help people"},
    ]

    print_success(f"Created {len(training_pairs)} training pairs")

    print_step(2, "Testing ContrastivePairSet creation")
    try:
        from wisent_guard.core import ContrastivePairSet

        pair_set = ContrastivePairSet.from_phrase_pairs(
            name="debug_training",
            phrase_pairs=training_pairs,
            task_type="safety_classification",
        )

        print_success("ContrastivePairSet created successfully")
        print(f"Pair set name: {pair_set.name}")
        print(f"Task type: {pair_set.task_type}")

        return pair_set

    except Exception as e:
        print_error(f"ContrastivePairSet creation failed: {e}")
        traceback.print_exc()
        return None


def test_8_classifier_training(pair_set, wisent_model, layer_obj):
    """Test 8: Classifier training with debugging."""
    print_section("TEST 8: CLASSIFIER TRAINING DEBUG")

    print_step(1, "Creating classifier")
    try:
        from wisent_guard.core import Classifier

        classifier = Classifier(
            model_type="logistic", device=CONFIG["device"], threshold=0.5
        )

        print_success("Classifier created successfully")
        print(f"Classifier type: {classifier.model_type}")
        print(f"Classifier device: {classifier.device}")
        print(f"Classifier threshold: {classifier.threshold}")

    except Exception as e:
        print_error(f"Classifier creation failed: {e}")
        traceback.print_exc()
        return None

    print_step(2, "Attempting classifier training with detailed debugging")
    try:
        pair_set.extract_activations_with_model(wisent_model, layer_obj)
        # Attempt training
        start_time = time.time()
        results = pair_set.train_classifier(classifier, layer_obj)
        train_time = time.time() - start_time

        print_success(f"Classifier training succeeded in {train_time:.2f}s")
        print(f"Results: {results}")

        return classifier, results

    except Exception as e:
        print_error(f"Classifier training failed: {e}")
        traceback.print_exc()
        return None, None


def run_all_tests():
    """Run all debugging tests in sequence."""
    print_section("WISENT GUARD DEBUGGING DEMO")
    print(f"Model: {CONFIG['model_name']}")
    print(f"Device: {CONFIG['device']}")
    print(f"Layer: {CONFIG['layer']}")
    print(f"Dtype: {CONFIG['dtype']}")

    # Test 1: Basic setup
    if not test_1_basic_setup():
        print_error("Basic setup failed, cannot continue")
        return

    # Test 2: Model loading
    model, tokenizer = test_2_model_loading()
    if model is None:
        print_error("Model loading failed, cannot continue")
        return

    # Test 3: Basic generation
    if not test_3_basic_generation(model, tokenizer):
        print_error("Basic generation failed, cannot continue")
        return

    # Test 4: Activation extraction
    activations = test_4_activation_extraction(model, tokenizer)
    if activations is None:
        print_error("Activation extraction failed, cannot continue")
        return

    # Test 5: Wisent Guard imports
    if not test_5_wisent_guard_imports():
        print_error("Wisent Guard imports failed, cannot continue")
        return

    # Test 6: Wisent Model wrapper
    wrapper_result = test_6_wisent_model_wrapper(model, tokenizer)
    if wrapper_result is None:
        print_error("Wisent Model wrapper failed, cannot continue")
        return

    wisent_model, layer_obj, wrapper_activations = wrapper_result

    # Test 7: Training data preparation
    pair_set = test_7_training_data_preparation()
    if pair_set is None:
        print_error("Training data preparation failed, cannot continue")
        return

    # Test 8: Classifier training
    classifier, results = test_8_classifier_training(pair_set, wisent_model, layer_obj)
    if classifier is None:
        print_error("Classifier training failed")
        return

    print_section("ALL TESTS COMPLETED SUCCESSFULLY!")
    print_success("Debugging demo completed without critical errors")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "wisent_model": wisent_model,
        "layer_obj": layer_obj,
        "classifier": classifier,
        "results": results,
    }


if __name__ == "__main__":
    try:
        components = run_all_tests()
        if components:
            print("\n🎉 Ready for performance benchmarking!")
        else:
            print("\n💥 Debugging found issues that need to be resolved")
    except KeyboardInterrupt:
        print("\n\n⏹️  Debugging interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error in debugging: {e}")
        traceback.print_exc()
