import json
import os
from pathlib import Path
from collections import defaultdict
import argparse

def find_grpo_json_files(root_dir: str):
    input_path_obj = Path(root_dir)
    grpo_files = list(input_path_obj.rglob('grpo_train_*.json'))
    print(f"Found {len(grpo_files)} 'grpo_train_*.json' files in '{root_dir}'.")
    return grpo_files

def process_grpo_files(input_dir_path_str: str, output_dir_path_str: str):
    output_dir = Path(output_dir_path_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory set to: {output_dir.resolve()}")

    aggregated_data_by_level = defaultdict(list)

    grpo_input_files = find_grpo_json_files(input_dir_path_str)

    if not grpo_input_files:
        print(f"No 'grpo_train_*.json' files found in '{input_dir_path_str}'. Nothing to process.")
        return

    for file_path in grpo_input_files:
        print(f"Processing file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                task_data_by_level = json.load(f)
            
            if not isinstance(task_data_by_level, dict):
                print(f"  Warning: Expected a dictionary at the top level of {file_path}, but got {type(task_data_by_level)}. Skipping this file.")
                continue

            for level_str, samples_list in task_data_by_level.items():
                if not level_str.isdigit():
                    print(f"  Warning: Found a non-numeric level key '{level_str}' in {file_path}. This key will be skipped.")
                    continue
                
                if not isinstance(samples_list, list):
                    print(f"  Warning: Expected a list of samples for level '{level_str}' in {file_path}, but got {type(samples_list)}. Skipping this level's data from this file.")
                    continue

                for i, sample in enumerate(samples_list):
                    try:
                        query = sample.get("query")
                        answer_actions = sample.get("answer_actions")

                        if query is None or not isinstance(query, dict):
                            print(f"  Warning: Sample {i} in {file_path} for level {level_str} is missing 'query' or 'query' is not a dict. Skipping sample.")
                            continue
                        if answer_actions is None or not isinstance(answer_actions, list):
                            print(f"  Warning: Sample {i} in {file_path} for level {level_str} is missing 'answer_actions' or it's not a list. Skipping sample.")
                            continue
                        
                        messages = query.get("messages")
                        images = query.get("images")

                        if messages is None or not isinstance(messages, list):
                            print(f"  Warning: Sample {i}'s query in {file_path} for level {level_str} is missing 'messages' or it's not a list. Skipping sample.")
                            continue
                        if images is None or not isinstance(images, list):
                             print(f"  Warning: Sample {i}'s query in {file_path} for level {level_str} is missing 'images' or it's not a list. Skipping sample.")
                             continue
                        
                        transformed_sample = {
                            "messages": messages,
                            "action": answer_actions, # "action" key in output corresponds to "answer_actions" from input
                            "images": images
                        }
                        aggregated_data_by_level[level_str].append(transformed_sample)
                    except Exception as e_sample:
                        print(f"  Error processing sample {i} in {file_path} for level {level_str}: {e_sample}. Sample snippet: {str(sample)[:200]}...")
                        
        except json.JSONDecodeError:
            print(f"  Error: Could not decode JSON from {file_path}. Skipping this file.")
        except Exception as e_file:
            print(f"  An unexpected error occurred while processing file {file_path}: {e_file}. Skipping this file.")

    if not aggregated_data_by_level:
        print("No data was successfully aggregated from any files. Exiting.")
        return

    print("\nWriting aggregated data to JSONL files by difficulty level...")
    max_level_found = 0
    levels_written = []

    sorted_levels = sorted(aggregated_data_by_level.keys(), key=lambda k: int(k) if k.isdigit() else float('inf'))

    for level_str in sorted_levels:
        if not level_str.isdigit(): # Should have been filtered, but as a safeguard
            continue
        
        level_int = int(level_str)
        if level_int > max_level_found:
            max_level_found = level_int
        
        output_file_path = output_dir / f"grpo_train_level{level_int}.jsonl"
        
        samples_for_this_level = aggregated_data_by_level[level_str]
        if not samples_for_this_level:
            print(f"  No samples aggregated for level {level_int}. File {output_file_path} will not be created.")
            continue

        try:
            with open(output_file_path, 'w', encoding='utf-8') as f_out:
                for sample_to_write in samples_for_this_level:
                    f_out.write(json.dumps(sample_to_write, ensure_ascii=False) + '\n')
            print(f"  Successfully wrote {len(samples_for_this_level)} samples to {output_file_path}")
            levels_written.append(level_int)
        except Exception as e_write:
            print(f"  Error writing to {output_file_path}: {e_write}")
        
    print(f"\nProcessing complete.")
    if levels_written:
        print(f"Output files for levels {sorted(levels_written)} are in: {output_dir.resolve()}")
        print(f"Maximum difficulty level found and processed with data: {max_level_found}")
    else:
        print(f"No data was written to any output files.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Cleans and aggregates GRPO JSON files (e.g., grpo_train_TASK.json) into JSONL format, "
                    "separating them by difficulty level (e.g., grpo_train_level1.jsonl)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/cluster/home1/wzx/EgoReasoner/data/egoreasoner", # Default as per your "Raw File" path
        help="Root directory containing the 'grpo_train_*.json' files to process. Script will search recursively."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/cluster/home1/wzx/EgoReasoner/data/data_/imitation", # Sensible default for output
        help="Directory where the 'grpo_train_level{N}.jsonl' files will be saved."
    )
    
    args = parser.parse_args()

    process_grpo_files(args.input_dir, args.output_dir)