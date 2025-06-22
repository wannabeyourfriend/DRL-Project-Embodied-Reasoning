import os
import sys
import json
import argparse
import time
import requests
import base64
import re
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from embodied_reasoner.evaluate.ai2thor_engine.RocAgent import RocAgent
    from ai2thor.controller import Controller
    from embodied_reasoner.api_keys_config import QWEN_API_KEY
except ImportError:
    print("Warning: Running with mock classes. AI2THOR environment is not active.")
    QWEN_API_KEY = "YOUR_DUMMY_API_KEY"
    class MockController:
        def __init__(self, **kwargs):
            self.last_event = type('event', (), {'metadata': {'inventoryObjects': []}})()
        def stop(self): print("MockController stopped.")
        def step(self, action_dict): # Add mock for new actions
             print(f"MockController received action: {action_dict}")
             return self.last_event
         
    class MockAgent:
        def __init__(self, **kwargs):
            self.controller = MockController()
            self.step_count = 0
        def _get_state(self):
            self.step_count += 1
            image_path = f'./mock_image_{self.step_count}.png'
            if not os.path.exists(image_path):
                with open(image_path, 'w') as f: f.write('mock image')
            navigations = ['Sofa', 'CoffeeTable', 'TVStand']
            interactions = ['Pillow', 'RemoteControl']
            return image_path, navigations, interactions
        def __getattr__(self, name):
            return lambda *args, **kwargs: self._get_state()

    RocAgent = MockAgent
    Controller = MockController



SYSTEM_PROMPT =  """You are an intelligent robot in a virtual home. Your task is to follow human instructions by selecting the best action at each step.

**Available Actions:**
You can choose one of the following actions. Pay close attention to the required arguments.

*Movement & Perception:*
- "navigate to <object>": Move to a large, static object (e.g., Fridge, Sofa). The <object> MUST be chosen from the 'Available Navigation Targets' list.
- "move forward": Move a short distance forward to get a closer view.
- "turn <direction>": Turn the robot's body. <direction> must be either "left" or "right". Use this to scan a room.
- "look <direction>": Tilt the robot's head camera. <direction> must be either "up" or "down". Use this to see objects on high shelves or on the floor.
- "observe": Get images from your left, right, and rear perspectives to understand your immediate surroundings.

*Object Interaction:*
- "pickup <object>": Pick up a smaller, movable object. The <object> MUST be from the 'Visible Interactable Objects' list and be within reach.
- "put <target_object>": Place the object you are holding onto a surface or into a container. The <target_object> MUST be from the 'Visible Interactable Objects' list.
- "open <object>": Open an object with a door or lid (e.g., Fridge, Cabinet). The <object> MUST be from the 'Visible Interactable Objects' list.
- "close <object>": Close an object that is currently open. The <object> MUST be from the 'Visible Interactable Objects' list.
- "toggle <object>": Turn an appliance on or off (e.g., Lamp, Television). The <object> MUST be from the 'Visible Interactable Objects' list.

*Task Completion:*
- "end": Use this action only when you are certain you have fully completed the task.

**CRITICAL RULES FOR YOUR RESPONSE:**
1.  **Mandatory Thinking**: Your response MUST contain a `<think>` section followed by an `<action>` section.
2.  **Explain Yourself**: In `<think>`, clearly explain your reasoning. Analyze the visual evidence, the task, your inventory, and your action history to justify your next move.
3.  **Strict Action Format**: In `<action>`, provide a SINGLE, VALID JSON object for your chosen action. Do not add any text outside the JSON.
    - Examples: `{"action": "navigate to", "object": "Fridge"}` or `{"action": "turn", "direction": "left"}` or `{"action": "end"}`
4.  **Use Provided Lists**: For any action that requires an object or target, you MUST select a valid name from the 'Available Navigation Targets' or 'Visible Interactable Objects' lists provided in the current state. DO NOT invent object names.
5.  **Break Loops**: Carefully review your 'Action History'. If you are stuck repeating the same actions (like 'observe' or 'turn') without making progress, you MUST change your strategy. Use actions like `move forward`, `look up/down`, or `Maps to` a different major object to explore new areas and find what you need.

**Example Response Format:**
<think>The task is to find a remote control. I can see a 'Sofa' in the navigation list. Sofas are common places for remotes. My previous 'observe' actions didn't reveal it, so instead of observing again, I will navigate to the Sofa to get a closer look at it and the nearby coffee table.</think>
<action>
{"action": "navigate to", "object": "Sofa"}
</action>
"""

INVALID_ACTION_PROMPT = """**ACTION FAILED**: The previous action you chose was **invalid or could not be executed**.

* **Your Failed Action**: `{failed_action_json}`
* **Reason for Failure**: The action was likely illegal. This happens when:
    1.  The target object for "navigate to" was not in the 'Available Navigation Targets' list.
    2.  The target for "pickup", "open", etc., was not in the 'Visible Interactable Objects' list, was not interactable (e.g., a wall), or was out of reach.
    3.  The action was contextually impossible (e.g., trying to "put" an item when your hand is empty).

Your state has **NOT** changed. You are seeing the exact same image and state information again.

**Current State (Unchanged)**:
- Step: {step}/{max_steps}
- Item in Hand: {holding_item}
- Available Navigation Targets: {legal_navigations}
- Visible Interactable Objects: {legal_interactions}
- Recent Action History: {action_history}

Please re-evaluate the situation. Analyze why your last action failed and choose a **different and valid** action from the available lists. Provide your new reasoning in `<think>` and the new action in `<action>`.
"""

# --- Helper Functions ---

def setup_logging(log_path):
    """Configures a logger to save to a file."""
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def encode_image_to_base64(image_path):
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 2. MODIFIED `call_llm_api` TO ACCEPT MULTIPLE IMAGES
def call_llm_api(api_key, system_prompt, user_prompt, logger, image_paths=None):
    """Calls the LLM API and logs the interaction. Can send multiple images."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    num_images = len(image_paths) if image_paths else 0
    full_prompt_for_log = f"---SYSTEM PROMPT---\n{system_prompt}\n\n---USER PROMPT---\n{user_prompt}"
    logger.info(f"--- OUTGOING PROMPT TO API (with {num_images} images) ---\n{full_prompt_for_log}\n--------------------")

    system_message = {"role": "system", "content": [{"text": system_prompt}]}
    
    user_content = []
    if image_paths and isinstance(image_paths, list):
        for path in image_paths:
            if os.path.exists(path):
                user_content.append({"image": f"data:image/jpeg;base64,{encode_image_to_base64(path)}"})
            else:
                logger.warning(f"Image path not found, skipping: {path}")
            
    user_content.append({"text": user_prompt})
    
    user_message = {"role": "user", "content": user_content}
    messages = [system_message, user_message]
    
    payload = {
        "model": "qwen-vl-max",
        "input": {"messages": messages},
        "parameters": {"result_format": "message"}
    }
    
    response_text_for_log = ""
    try:
        response = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        response_text_for_log = response.text
        response_data = response.json()
        
        choice = response_data["output"]["choices"][0]
        message = choice["message"]
        
        action_text = message["content"][0]["text"]
        reason_text = message.get("reasoning_content", "No reasoning content provided by API.")
        
        return {"action_text": action_text, "reason_text": reason_text}

    except Exception as e:
        error_message = f"API Call Failed: {e}"
        server_response = response_text_for_log if response_text_for_log else "No response text available."
        logger.error(f"--- API CALL FAILED ---\n{error_message}\n--- SERVER RESPONSE ---\n{server_response}\n--------------------")
        print(f"{error_message}\nServer Response: {server_response}")
        return None
    finally:
        if response_text_for_log:
            logger.info(f"--- INCOMING RAW RESPONSE FROM API ---\n{response_text_for_log}\n--------------------")

def parse_llm_response(api_output_dict):
    """Parses the structured output from the API call."""
    if not api_output_dict:
        return {"think": "Error: No data from API call.", "action_json": None}
    
    think_content = api_output_dict.get('reason_text', 'No thoughts provided.')
    action_text = api_output_dict.get('action_text', '')

    action_match = re.search(r'<action>(.*?)</action>', action_text, re.DOTALL)
    action_json = None
    if action_match:
        action_str = action_match.group(1).strip()
        try:
            action_json = json.loads(action_str)
        except json.JSONDecodeError:
            think_content += f"\n[Parsing Error: Failed to decode JSON from action block: '{action_str}']"
    
    return {"think": think_content, "action_json": action_json}



def main():
    parser = argparse.ArgumentParser(description='API-based autonomous decision-making demo.')
    parser.add_argument('--scene', type=str, default='FloorPlan203', help='AI2THOR scene name.')
    parser.add_argument('--save_path', type=str, default='./data/api_demo/', help='Save path for data and logs.')
    parser.add_argument('--task_id', type=int, default=0, help='Task ID.')
    parser.add_argument('--platform_type', type=str, default='GPU', help='Platform type.')
    parser.add_argument('--task', type=str, default='Find a remote and put it on the coffee table.', help='Task description.')
    parser.add_argument('--max_steps', type=int, default=20, help='Maximum number of steps.')
    args = parser.parse_args()
    
    NUM_HISTORICAL_IMAGES = 3
    MAX_RETRIES_PER_STEP = 2
    
    os.makedirs(args.save_path, exist_ok=True)
    log_file_path = os.path.join(args.save_path, 'api_communication.log')
    setup_logging(log_file_path)
    logger = logging.getLogger()
    logger.info("--- Starting New Run ---")
    print(f"Logging API interactions to: {log_file_path}")

    print(f"Initializing AI2THOR controller, scene: {args.scene}")
    controller = Controller(scene=args.scene, gridSize=0.25, width=640, height=480, fieldOfView=90, renderDepthImage=True)
    
    target_objects = ["RemoteControl|1", "CoffeeTable|1"]
    related_objects = ["CoffeeTable|1"]
    navigable_objects = ["CounterTop", "Sink", "Fridge", "DiningTable", "Chair", "CoffeeTable", "Sofa", "TVStand"]
    
    print("Initializing RocAgent...")
    agent = RocAgent(
        controller=controller, save_path=args.save_path, scene=args.scene,
        visibilityDistance=1.5, gridSize=0.25, fieldOfView=90,
        target_objects=target_objects, related_objects=related_objects,
        navigable_objects=navigable_objects, taskid=args.task_id,
        platform_type=args.platform_type
    )
    
    print("\n1. Initializing agent position")
    image_fp, legal_navigations, legal_interactions = agent.init_agent_corner()
    print(f"Initialization complete. Image saved to: {image_fp}")

    action_history = []
    
    for step in range(args.max_steps):
        print(f"\n===== STEP {step+1} =====")
        
        inventory = agent.controller.last_event.metadata["inventoryObjects"]
        holding_item = inventory[0]["objectType"] if inventory else "Nothing"
        
        num_total_images = min(step, NUM_HISTORICAL_IMAGES) + 1
        user_prompt = f"""You have been provided with {num_total_images} images. The last image is your most current view, and the preceding ones are from your recent history to provide context.

Task: {args.task}

Current State:
- Step: {step+1}/{args.max_steps}
- Item in Hand: {holding_item}
- Available Navigation Targets: {legal_navigations}
- Visible Interactable Objects: {legal_interactions}

Action History:
{json.dumps(action_history, indent=2)}

Review the 'CRITICAL RULES' from the system prompt. Based on all the visual information and your state, provide your detailed thought process in <think> and your next action in <action>.
"""
        
        print("Calling LLM API for a decision...")

        historical_image_paths = [h['image_path'] for h in action_history[-NUM_HISTORICAL_IMAGES:]]
        all_image_paths = historical_image_paths + [image_fp]

        api_output_dict = call_llm_api(QWEN_API_KEY, SYSTEM_PROMPT, user_prompt, logger, image_paths=all_image_paths)
        
        parsed_response = parse_llm_response(api_output_dict)
        think_process = parsed_response["think"]
        decision_json = parsed_response["action_json"]
        
        print("\n--- LLM Reasoning ---")
        print(think_process)
        print("--- LLM Action ---")
        print(json.dumps(decision_json, indent=2) if decision_json else "No valid action JSON was parsed.")
        print("--------------------\n")

        if not decision_json or "action" not in decision_json:
            print("Could not parse a valid action. Defaulting to 'observe'.")
            action_result = agent.observe()
            action_history.append({"step": step+1, "action": "observe", "object": "", "thought": "Failed to parse API response.", "image_path": image_fp})
            if action_result: image_fp, legal_navigations, legal_interactions = action_result
            continue

        try:
            action = decision_json.get("action", "")
            object_name = decision_json.get("object", "")
            
            print(f"Executing Decision: Action='{action}', Object='{object_name}'")
            
            action_history.append({
                "step": step+1, 
                "thought": think_process,
                "action": action, 
                "object": object_name,
                "image_path": image_fp 
            })
            
            action_result = None
            if action == "end":
                print("Task marked as 'end' by the model.")
                break
            elif action == "observe":
                action_result = agent.observe()
            elif action == "move forward":
                action_result = agent.move_forward(0.5)
            elif object_name:
                if action == "navigate to": action_result = agent.navigate(object_name)
                elif action == "pickup": action_result = agent.pick_up(object_name)
                elif action == "put in": action_result = agent.put_in(object_name)
                elif action == "toggle": action_result = agent.toggle(object_name)
                elif action == "open": action_result = agent.open(object_name)
                elif action == "close": action_result = agent.close(object_name)
                else:
                    print(f"Unknown action: '{action}'. Defaulting to 'observe'.")
                    action_result = agent.observe()
            else:
                 print(f"Action '{action}' requires an object, but none was provided. Defaulting to 'observe'.")
                 action_result = agent.observe()

            if action_result is None:
                print(f"Agent action '{action}' on '{object_name}' failed internally. Observing for safety.")
                logger.warning(f"Agent action '{action}' on '{object_name}' returned None.")
                image_fp, legal_navigations, legal_interactions = agent.observe()
            else:
                image_fp, legal_navigations, legal_interactions = action_result
                print(f"Action executed. New image saved to: {image_fp}")
            
            with open(os.path.join(args.save_path, "action_history.json"), "w", encoding="utf-8") as f:
                json.dump(action_history, f, ensure_ascii=False, indent=4)
                
            time.sleep(1)

        except Exception as e:
            print(f"An unexpected error occurred during action execution: {e}")
            logger.error(f"Unexpected error for action '{action}' with object '{object_name}': {e}", exc_info=True)
            print("Defaulting to 'observe' action.")
            image_fp, legal_navigations, legal_interactions = agent.observe()
    
    logger.info("--- Run Finished ---")
    print("\n===== DEMO FINISHED =====")
    print(f"All images and data saved to: {args.save_path}")
    
    controller.stop()

if __name__ == "__main__":
    main()