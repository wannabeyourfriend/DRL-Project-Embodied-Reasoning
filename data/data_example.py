action_history = ["observe"]
available_actions = ["navigate to <object>", "pickup <object>", "put in", "toggle", "open"]
available_objects = ["Apple", "DiningTable", "Fridge"]

dataset = \
{ 
    "images":[  # to handle partially observable environments
        "observe_image_path"
    ],
    "message":[
        {
            "role": "user",
            "content": \
f"""
Given history observation and action history {",".join(action_history)}, what is your planning based on the current state? Available actions are {", ".join(available_actions)}. Available objects are {", ".join(available_objects)}.

Strictly follow the format:
<think>
reasoning process here
</think>
<action>
[action1, action2, ...]
</action>
""".strip()
        }
    ],
    "env_config": { # env configs and success metrics, for simulator calling in reward function
        "scene": "FloorPlan1",
        "task_id": "0",
        "platform_type": "GPU",
        "task": 'Find a remote and put it on the coffee table.',
        "max_steps": 20
    }
}
