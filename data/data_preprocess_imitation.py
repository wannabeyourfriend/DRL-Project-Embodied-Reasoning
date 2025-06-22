import json
import re
from pathlib import Path
from collections import defaultdict
import os # For path normalization (though Pathlib is used more)
import argparse # Import argparse

USER_INPUT_FILE_CONST = "/nfs/home1/wzx/EgoReasoner/data/embodied_reasoner/train_multiturn_9390.json"
BASE_OUTPUT_DIR_IMITATION_TEMPLATE = "/nfs/home1/wzx/EgoReasoner/data/egoreasoner/imitation/{task_string}/"
BASE_OUTPUT_DIR_SFT_TEMPLATE = "/nfs/home1/wzx/EgoReasoner/data/egoreasoner/sft/{task_string}/"
BASE_IMAGE_PREFIX_TEMPLATE = "/nfs/home1/wzx/EgoReasoner/data/images/{task_string}/" # Ensure this ends with a slash if it's a directory prefix

def extract_action_from_content(content: str):
    match = re.search(r"<DecisionMaking>(.*?)</DecisionMaking>", content)
    if match:
        return match.group(1)
    return None

def prepare_datasets(input_file_path: str, output_dir_sft: str, image_prefix_filter: str, task_name_for_file: str):
    input_path = Path(input_file_path)
    output_path_base = Path(output_dir_sft)
    output_path_base.mkdir(parents=True, exist_ok=True)

    sft_output_file = output_path_base / f"sft_train_{task_name_for_file}.json"
    grpo_output_file = output_path_base / f"grpo_train_{task_name_for_file}.json"

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            all_trajectories_raw = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_path}")
        return

    sft_filtered_trajectories = []
    relevant_trajectories_for_grpo = []

    print(f"Filtering trajectories with image prefix: {image_prefix_filter}")
    
    normalized_filter_prefix_str = Path(image_prefix_filter).resolve().as_posix()
    if not normalized_filter_prefix_str.endswith('/'):
        normalized_filter_prefix_str += '/'

    for trajectory_raw in all_trajectories_raw:
        images_in_trajectory = trajectory_raw.get("images", [])
        if not images_in_trajectory:
            continue

        all_match_prefix = True
        for img_path_str in images_in_trajectory:
            resolved_img_path_str = Path(img_path_str).resolve().as_posix()
            if not resolved_img_path_str.startswith(normalized_filter_prefix_str):
                all_match_prefix = False
                break
        
        if all_match_prefix:
            sft_filtered_trajectories.append(trajectory_raw)
            relevant_trajectories_for_grpo.append(trajectory_raw)

    with open(sft_output_file, 'w', encoding='utf-8') as f:
        json.dump(sft_filtered_trajectories, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(sft_filtered_trajectories)} SFT trajectories to {sft_output_file}")

    if not relevant_trajectories_for_grpo:
        print(f"No trajectories found matching the prefix '{image_prefix_filter}'. GRPO file will be empty.")
        with open(grpo_output_file, 'w', encoding='utf-8') as f:
            json.dump({}, f, indent=4, ensure_ascii=False) # Empty object for GRPO
        print(f"Saved empty GRPO data to {grpo_output_file}")
        return

    grpo_data_by_difficulty = defaultdict(list)

    for trajectory in relevant_trajectories_for_grpo:
        messages = trajectory["messages"]
        images = trajectory["images"] # list of image paths for the trajectory

        if not messages or messages[0]["role"] != "system":
            print(f"Warning: Trajectory {trajectory.get('id', 'N/A')} - First message is not system prompt or no messages. Skipping.")
            continue
        system_message_obj = messages[0]

        dialog_turns_data = []
        extracted_actions = []
        num_interactions = len(images) # Number of image-based interaction steps

        valid_trajectory = True
        for i in range(num_interactions): # Loop for each image, expecting a user and assistant turn
            user_turn_idx = 2 * i + 1
            assistant_turn_idx = 2 * i + 2

            if not (user_turn_idx < len(messages) and assistant_turn_idx < len(messages)):
                print(f"Warning: Trajectory {trajectory.get('id', 'N/A')} turn {i} - Message index out of bounds. Expected at least {assistant_turn_idx + 1} messages, got {len(messages)}. Skipping.")
                valid_trajectory = False
                break
            
            user_message_obj = messages[user_turn_idx]
            assistant_message_obj = messages[assistant_turn_idx]
            image_path = images[i] # image_path for the i-th interaction

            if user_message_obj["role"] != "user" or assistant_message_obj["role"] != "assistant":
                print(f"Warning: Trajectory {trajectory.get('id', 'N/A')} turn {i} - Unexpected role sequence. Skipping.")
                valid_trajectory = False
                break
            
            action_str = extract_action_from_content(assistant_message_obj["content"])
            if action_str is None:
                print(f"Warning: Trajectory {trajectory.get('id', 'N/A')} turn {i} - Could not extract action. Content: {assistant_message_obj['content'][:100]}... Skipping.")
                valid_trajectory = False
                break

            dialog_turns_data.append({
                "user_message": user_message_obj,
                "assistant_message": assistant_message_obj,
                "image_path": image_path,
                "action": action_str
            })
            extracted_actions.append(action_str)
        
        if not valid_trajectory or not dialog_turns_data:
            continue

        num_total_actions_in_trajectory = len(extracted_actions)
        if num_total_actions_in_trajectory == 0:
            continue
        
        for d_difficulty in range(1, num_total_actions_in_trajectory + 1):
            num_actions_to_predict = d_difficulty
            idx_first_action_in_target = num_total_actions_in_trajectory - num_actions_to_predict
            
            target_actions_list = extracted_actions[idx_first_action_in_target : idx_first_action_in_target + num_actions_to_predict]

            query_messages = [system_message_obj]
            query_images = []

            for k_context_turn in range(idx_first_action_in_target):
                turn_data = dialog_turns_data[k_context_turn]
                query_messages.append(turn_data["user_message"])
                query_messages.append(turn_data["assistant_message"])
                query_images.append(turn_data["image_path"])
            
            current_observation_turn_data = dialog_turns_data[idx_first_action_in_target]
            query_messages.append(current_observation_turn_data["user_message"])
            query_images.append(current_observation_turn_data["image_path"])

            grpo_sample = {
                "query": {
                    "messages": query_messages,
                    "images": query_images
                },
                "answer_actions": target_actions_list
            }
            grpo_data_by_difficulty[str(d_difficulty)].append(grpo_sample)

    with open(grpo_output_file, 'w', encoding='utf-8') as f:
        json.dump(grpo_data_by_difficulty, f, indent=4, ensure_ascii=False)
    print(f"Saved GRPO data to {grpo_output_file}")
    if grpo_data_by_difficulty:
        for difficulty_level, samples in sorted(grpo_data_by_difficulty.items(), key=lambda item: int(item[0])):
            print(f"  GRPO Difficulty {difficulty_level}: {len(samples)} samples")
    else:
        print(f"  No GRPO samples generated for this task.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare SFT and GRPO datasets for a specific task type.")
    parser.add_argument("task_string", type=str, help="The task-specific string (e.g., 'navigate1open1pickup0') used to determine output paths and image filtering.")
    args = parser.parse_args()
    task_string = args.task_string
    user_output_dir_imitation = BASE_OUTPUT_DIR_IMITATION_TEMPLATE.format(task_string=task_string)
    user_output_dir_sft = BASE_OUTPUT_DIR_SFT_TEMPLATE.format(task_string=task_string)
    user_image_prefix = BASE_IMAGE_PREFIX_TEMPLATE.format(task_string=task_string)
    
    user_input_file = USER_INPUT_FILE_CONST

    print(f"\n--- Running for task: {task_string} ---")
    print(f"Using Input File: {user_input_file}")
    print(f"Target SFT/GRPO Output Directory: {user_output_dir_sft}")
    print(f"Target Imitation Output Directory: {user_output_dir_imitation}")
    print(f"Filtering by Image Prefix: {user_image_prefix}")
    print(f"--------------------------------------------------")
    
    Path(user_output_dir_sft).mkdir(parents=True, exist_ok=True)
    Path(user_output_dir_imitation).mkdir(parents=True, exist_ok=True) # Create imitation dir as well

    prepare_datasets(user_input_file, user_output_dir_sft, user_image_prefix, task_string)
    print(f"--- Finished processing for task: {task_string} ---")