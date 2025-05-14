import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# --- Configuration ---
# Construct the absolute path to your model directory
# Assumes this script is in ~/wangzixuan/src/ and Qwenmodel is in ~/wangzixuan/
base_dir = os.path.expanduser("~/wangzixuan") # Gets /cluster/home2/yueyang/wangzixuan
model_name_or_path = os.path.join(base_dir, "Qwen2.5-Omni-7B")

print(f"Attempting to load model from: {model_name_or_path}")

# --- Check if model path exists ---
if not os.path.isdir(model_name_or_path):
    print(f"ERROR: Model directory not found at {model_name_or_path}")
    print("Please ensure the path is correct and the model files are present.")
    exit()

try:
    # --- 1. Load Tokenizer ---
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True  # Important for Qwen models with custom code
    )
    print("Tokenizer loaded successfully!")
    print(f"Tokenizer class: {tokenizer.__class__}")

    # --- 2. Load Model ---
    print("\nLoading model...")
    # For a 7B model, using "auto" for dtype and device_map is highly recommended.
    # This will try to use bfloat16/float16 and available GPUs.
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",       # Automatically selects best dtype (e.g., bfloat16, float16)
        device_map="auto",        # Automatically maps to available GPU(s) or CPU
        trust_remote_code=True    # Important for Qwen models
    )
    print("Model loaded successfully!")
    print(f"Model class: {model.__class__}")
    print(f"Model is on device: {model.device}") # Shows which device (e.g., cuda:0, cpu)

    # --- 3. Minimal Inference Test (Optional, but good to check) ---
    print("\nPerforming a minimal generation test...")
    # Prepare a simple prompt using the tokenizer's chat template if available
    # (Qwen models typically have one)
    messages = [{"role": "user", "content": "Hello!"}]
    try:
        text_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e_template:
        print(f"Could not apply chat template (this might be okay for some models): {e_template}")
        print("Using a raw prompt instead.")
        text_prompt = "Hello!"

    inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)

    # Generate a few tokens
    with torch.no_grad(): # Crucial for inference
        outputs = model.generate(**inputs, max_new_tokens=10)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Minimal generation output: {generated_text}")

    print("\n--- Test Completed Successfully! ---")

except ImportError as e:
    print(f"\n--- IMPORT ERROR ---")
    print(f"Error: {e}")
    print("This usually means a required library (like 'transformers' or 'torch') is not installed correctly,")
    print("or the 'transformers' library version is too old for this model.")
    print("Please ensure you are in the correct Conda environment (qwen7bomini) and have run:")
    print("  pip install --upgrade transformers torch accelerate sentencepiece")
    print(f"Current transformers path seems to be: {e.path}" if hasattr(e, 'path') else "")


except Exception as e:
    print(f"\n--- AN ERROR OCCURRED ---")
    print(f"Error: {e}")
    print("Troubleshooting tips:")
    print("1. Ensure you have enough GPU memory (VRAM) for a 7B model.")
    print("2. Check if the model files in 'Qwenmodel' are complete and not corrupted.")
    print("3. If it's an Out Of Memory (OOM) error, you might need a GPU with more VRAM or explore model quantization.")