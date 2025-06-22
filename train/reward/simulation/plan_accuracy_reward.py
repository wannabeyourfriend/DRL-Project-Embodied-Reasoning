import os
import re
import warnings
import json
from typing import Dict, List, Union, Optional

from swift.plugin.orm import ORM
from env_checker import EnvChecker

# Global dictionary for registering reward functions
orms = {}


class PlanAccuracy(ORM):
    """
    Plan accuracy reward function, used to evaluate whether the model's answer can result in successful execution in AI2THOR environment.
    """

    # Class constants for identifying answers
    PLAN_PATTERN = r'\[\s*"(?:[^"]*)"(?:\s*,\s*"(?:[^"]*)")*\s*\]'

    # 匹配以 [ 开头、] 结尾，内部为 {"action": "...", "object": "..."} 的一个或多个字典
    # PLAN_PATTERN = (
    #     r'\[\s*'
    #     r'(?:\{\s*"action"\s*:\s*"[^"]+"\s*,\s*"object"\s*:\s*"[^"]+"\s*\}\s*,?\s*)+'
    #     r'\]'
    # )

    def __init__(self):
        """
        Initialize the plan accuracy reward function
        """
        self.format_weight = 1.0  # Weight for format correctness
        self.length_weight = 1.0  # Weight for length of the plan
        self.execution_weight = 1.0  # Weight for successful execution in the environment

    def normalize_plan(self, answer: str) -> Union[List[str], float]:
        """
        Normalize answer format, extract answer from text
        The answer should be like 
        plan = [
            {"action": "navigate to", "object": "Sofa"}, 
            {"action": "navigate to", "object": "Apple"}
        ]
        """
        answer = answer.strip()
        
        try:
            plan_matches = list(re.finditer(self.PLAN_PATTERN, answer, re.IGNORECASE))
            if plan_matches:
                # Use the last matched answer
                plan_str = plan_matches[-1].group(0)
                # json_plan_str = re.sub(r"\}\s*\{", "}, {", raw_plan)
                plan = json.loads(plan_str)
                assert isinstance(plan, list), "The plan should be a list of actions."
                format_reward = 1.0  # Reward for correct format
                return plan, format_reward
            else:
                format_reward = 0.0
                return answer, format_reward
        except json.JSONDecodeError as e: # other exceptions? -> add when debugging
            format_reward = 0.0
            return answer, format_reward
    
    def length_reward(self, plan: List[str]) -> float:
        """
        Calculate a reward based on the length of the plan.
        Shorter plans are generally preferred, so we return a negative value for longer plans.
        """
        if not isinstance(plan, list) or not plan:
            return -1.0 # a penalty for empty or invalid plans
        # Reward is inversely proportional to the length of the plan
        return -len(plan) * 0.05
    
    def execution_reward(self, plan: List[str], env_config: Dict) -> float:
        """
        Call a external environment checker to determine if the plan can be executed successfully.
        """
        env_checker = EnvChecker(env_config)
        info = env_checker.check(plan)
        execution_reward = 1.0 if info["success"] else 0.0
        del env_checker  # Clean up to free resources
        return execution_reward

    def __call__(self, completions, env_config, **kwargs):
        """
        Calculate the reward based on the model's plan and the expected solution.
        """
        rewards = []

        # Ensure both completions and env_config are lists
        if not isinstance(completions, list):
            completions = [completions]
        if not isinstance(env_config, list):
            env_config = [env_config]

        # calculate diffrent types of rewards for each pair
        for completion, env_cfg in zip(completions, env_config):
            plan, format_reward = self.normalize_plan(completion)
            if isinstance(plan, str):
                rewards.append(-3.0)
                continue
            
            length_reward = self.length_reward(plan)
            execution_reward = self.execution_reward(plan, env_cfg)
            
            # Combine all rewards with their respective weights
            reward = self.format_weight * format_reward
            reward += self.length_weight * length_reward
            reward += self.execution_weight * execution_reward
            
            rewards.append(reward)
        
        return rewards

# Register reward function
orms['plan_accuracy'] = PlanAccuracy

# For compatibility, also register under the name 'plan_accuracy'
from swift.plugin.orm import orms
orms['plan_accuracy'] = PlanAccuracy


if __name__ == "__main__":
    # Example usage of the PlanAccuracy reward function
    plan_accuracy = PlanAccuracy()
    completion = """
<think>
    The task is to find the Apple and put it on the DiningTable.
    The available actions are to navigate to the Apple, pick it up, and then put it on the DiningTable.
    [
        "navigate to Apple", 
        "pick up Apple", 
        "end"
    ]
</think>
<answer>
    [
        "navigate to Apple", 
        "move forward",
        "pick up Apple", 
        "put in DiningTable",
        "end"
    ]
</answer>
"""
    plan, format_reward = plan_accuracy.normalize_plan(completion)
    length_reward = plan_accuracy.length_reward(plan)
    print(f"Normalized plan:")
    print(plan)
    print(f"Format reward: {format_reward}")
    print(f"Length reward: {length_reward}")