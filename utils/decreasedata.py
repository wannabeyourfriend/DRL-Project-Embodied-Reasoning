import json
import re

# Renamed constant for clarity when appending
HISTORY_APPENDIX_PREFIX = "\n\nPrevious actions taken: "

def extract_actions_from_assistant(content: str) -> list[str]:
    """Extracts actions from <DecisionMaking> tags in assistant's content."""
    actions = re.findall(r"<DecisionMaking>(.*?)</DecisionMaking>", content)
    return [action.strip() for action in actions]

def clean_grpo_trajectory_messages(data: dict) -> dict:
    """
    Cleans the 'messages' field of a single GRPO trajectory entry.
    - Keeps the first system prompt.
    - Keeps the first user prompt.
    - Extracts all historical actions from assistant messages.
    - Appends historical actions to the content of the first user prompt.
    """
    if "messages" not in data or not isinstance(data["messages"], list):
        return data

    original_messages = data["messages"]
    new_messages = []

    first_system_prompt = None
    first_user_prompt = None # This will store the dictionary of the first user message
    historical_actions = []

    for message in original_messages:
        role = message.get("role")
        content = message.get("content")

        if role == "system" and first_system_prompt is None:
            first_system_prompt = message
        elif role == "user" and first_user_prompt is None:
            # Store the entire first user message dictionary
            first_user_prompt = message # Make sure to copy if modifying in place and original is needed elsewhere
        elif role == "assistant" and content:
            actions_found = extract_actions_from_assistant(content)
            if actions_found:
                historical_actions.extend(actions_found)

    # Construct the new messages list
    if first_system_prompt:
        new_messages.append(first_system_prompt)

    if first_user_prompt:
        # Create a mutable copy if you plan to modify it,
        # especially if original_messages could be reused.
        # For this script's flow, directly modifying is fine if first_user_prompt is unique per call.
        current_first_user_prompt = {"role": first_user_prompt["role"], "content": first_user_prompt["content"]}


        if historical_actions:
            history_string = HISTORY_APPENDIX_PREFIX + ", ".join(historical_actions)
            # Append the history string to the content of the first user prompt
            current_first_user_prompt["content"] = f"{current_first_user_prompt['content']}{history_string}"

        new_messages.append(current_first_user_prompt)

    cleaned_data = {key: value for key, value in data.items() if key != "messages"}
    cleaned_data["messages"] = new_messages

    return cleaned_data

def process_jsonl_file(input_filepath: str, output_filepath: str):
    """
    Reads a JSONL file, processes each line using clean_grpo_trajectory_messages,
    and writes the cleaned data to a new JSONL file.
    """
    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile, \
             open(output_filepath, 'w', encoding='utf-8') as outfile:
            for line_number, line in enumerate(infile, 1):
                try:
                    original_data = json.loads(line.strip())
                    cleaned_data = clean_grpo_trajectory_messages(original_data)
                    outfile.write(json.dumps(cleaned_data) + '\n')
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON on line {line_number} in {input_filepath}")
                except Exception as e:
                    print(f"Warning: Error processing line {line_number} in {input_filepath}: {e}")
        print(f"Successfully processed '{input_filepath}' and saved to '{output_filepath}'")
    except FileNotFoundError:
        print(f"Error: Input file '{input_filepath}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Example Usage (modify input_file and output_file as needed) ---
if __name__ == "__main__":
    # Define your input and output file paths here
    # For example, to run it on the files you mentioned previously:
    input_file = "/cluster/home1/wzx/EgoReasoner/data/full/embodied_agent_train_dataset_last_image.jsonl"
    output_file = "/cluster/home1/wzx/EgoReasoner/data/imitation/embodied_agent_train_dataset_last_image.jsonl" # new output name

    # # Or use a dummy example for quick testing:
    # dummy_input_data = [
    #     {"messages": [{"role": "system", "content": "System prompt 1."}, {"role": "user", "content": "User prompt 1 initial content."}, {"role": "assistant", "content": "Thinking... <DecisionMaking>Action1</DecisionMaking> Then more thinking."}, {"role": "user", "content": "User prompt 2 (should be ignored for first_user_prompt)."}, {"role": "assistant", "content": "Okay. <DecisionMaking>Action2</DecisionMaking>"}], "action": ["pickup A"], "images": ["img1.png"]},
    #     {"messages": [{"role": "user", "content": "Only user prompt, no system."}, {"role": "assistant", "content": "<DecisionMaking>SoloAction</DecisionMaking>"}], "action": ["end"], "images": []}
    # ]
    # input_file = "sample_grpo_input_for_consolidated_history.jsonl"
    # output_file = "cleaned_grpo_output_consolidated_history.jsonl"

    # with open(input_file, 'w', encoding='utf-8') as f:
    #     for item in dummy_input_data:
    #         f.write(json.dumps(item) + '\n')
    # print(f"Created dummy input file: {input_file}")

    process_jsonl_file(input_file, output_file)

    # print(f"\n--- Contents of '{output_file}' (first line if multiple) ---")
    # try:
    #     with open(output_file, 'r', encoding='utf-8') as f:
    #         processed_line = json.loads(f.readline().strip())
    #         print(json.dumps(processed_line, indent=2)) # Pretty print the first processed line
    # except FileNotFoundError:
    #     print(f"Output file {output_file} was not created.")
    # except json.JSONDecodeError:
    #     print("Could not parse JSON from output file.")