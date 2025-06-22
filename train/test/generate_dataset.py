
import json
import random

def generate_example():
    
    a = random.randint(0, 99_999)
    b = random.randint(0, 99_999)
    question = f"what is the answer for {a} + {b} ?"
    solution = str(a + b)

    example = {
        "messages": [
            {"role": "system", "content": "You are a helpful agent. Your final answer must strictly follow format: <think> ··· </think>\n<answer> ··· </answer>, for example, <think> thinking content </think>\n<answer> answer content </answer>."},
            {"role": "user",   "content": question},
            {"role": "assistant", "content": ""}
        ],
        "solution": solution
    }
    return example

def main(output_path: str, N: int = 1000):
    with open(output_path, "w", encoding="utf-8") as f:
        for _ in range(N):
            example = generate_example()
            f.write(json.dumps(example, ensure_ascii=False))
            f.write("\n")
    print(f"Generated {N} examples and saved to {output_path}")

if __name__ == "__main__":
    OUTPUT_FILE = "test_dataset.jsonl"
    NUM_EXAMPLES = 1000

    main(OUTPUT_FILE, NUM_EXAMPLES)
