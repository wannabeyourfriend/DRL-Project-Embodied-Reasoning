# Place this in a Python file, e.g., custom_agent_dataset.py
# This file can also be part of your --external_plugins

from swift.llm import register_dataset, DatasetMeta, SubsetDataset, ResponsePreprocessor
from typing import Dict, Any, List
import os # If you need to resolve relative image paths

class EmbodiedAgentPreprocessor(ResponsePreprocessor):
    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # 'messages' from your JSONL is already in a good format for model input.
        # 'images' from your JSONL is also in the expected list-of-paths format.
        
        # The 'action' field in your data is the ground truth sequence of actions.
        # We'll map this to 'solution' for the ORM.
        # The model will generate a 'completion' (its predicted action sequence).
        
        processed_row = {
            "messages": row.get("messages", []),
            "images": row.get("images", []), # Ensure image paths are accessible
            "solution": row.get("action", []) # This is your ground truth plan
        }
        
        # Optional: If your image paths are relative, you might need to make them absolute
        base_image_path = "/cluster/home1/wzx/EgoReasoner/data/imitation" # Configure this
        processed_row["images"] = [os.path.join(base_image_path, img_path.lstrip("./")) 
                                   for img_path in processed_row["images"]]

        # The GRPO trainer will use 'messages' and 'images' as input to the model.
        # The 'solution' field will be passed to your custom reward function.
        # The tutorial mentions:
        # `{'role': 'assistant', 'content': '<answer> 3 </answer>'}` will be removed by GRPOTrainer.
        # In your case, you don't have an assistant message in the input `messages`.
        # The model will generate the assistant's response (the plan).
        return processed_row


def register_my_embodied_agent_datasets():
    """Registers your custom embodied agent dataset."""
    dataset_files = {
        "train": "/cluster/home1/wzx/EgoReasoner/data/imitation/embodied_agent_train_dataset.jsonl",
        # "validation": "path/to/your/validation.jsonl" if you have one
    }

    # Check if files exist
    for split, file_path in dataset_files.items():
        if not os.path.exists(file_path):
            print(f"Warning: Dataset file for split '{split}' not found at '{file_path}'. Skipping registration for this split.")
            # Decide if you want to raise an error or just skip
            # raise FileNotFoundError(f"Dataset file for split '{split}' not found at '{file_path}'")

    # Only register datasets for which files exist
    subsets = []
    if os.path.exists(dataset_files.get("train")):
        subsets.append(SubsetDataset(name='default_train', subset_id=dataset_files["train"], split=['train']))
    # if os.path.exists(dataset_files.get("validation")):
    #     subsets.append(SubsetDataset(name='default_val', subset_id=dataset_files["validation"], split=['validation']))

    if not subsets:
        print("No dataset files found. Skipping registration of 'my_embodied_agent_dataset'.")
        return

    register_dataset(
        DatasetMeta(
            # A unique name for your dataset group
            dataset_id='my_embodied_agent_dataset', 
            # Path to the JSONL file(s)
            # For local files, you can use a dictionary mapping subset_id to actual file paths
            file_mapping={
                dataset_files[split]: dataset_files[split] 
                for split in dataset_files if os.path.exists(dataset_files[split])
            },
            subsets=subsets,
            preprocess_func=EmbodiedAgentPreprocessor(),
            # Tags are optional metadata
            tags=['embodied_agent', 'planning', 'multimodal'] 
        )
    )

# Call the registration function when this module is loaded
# register_my_embodied_agent_datasets()
# Note: SWIFT typically handles dataset registration through its mechanisms.
# You might need to ensure this code is run at an appropriate time,
# or adapt to how SWIFT expects custom datasets to be added.
# Often, placing this in the --external_plugins file and ensuring it's imported works.