import os
import re
import warnings
from typing import Dict, List, Union, Optional

from swift.plugin.orm import ORM

orms = {}


class TestAccuracy(ORM):

    def __init__(self):
        """
        Initialize the test accuracy reward function
        """
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")

    def extract_think_and_answer(text: str):
        """
        extract <think>...</think> and <answer>...</answer>
        return (think_content, answer_content)
        """
        think_match = re.search(r'<think>(.*?)</think>', text, re.S)
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.S)

        think_content = think_match.group(1).strip() if think_match else None
        answer_content = answer_match.group(1).strip() if answer_match else None

        return think_content, answer_content


    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []

        if not isinstance(completions, list):
            completions = [completions]

        if not isinstance(solution, list):
            solution = [solution] * len(completions)

        for content, sol in zip(completions, solution):
            try:
                think_content, answer_content = self.extract_think_and_answer(content)
                if sol in answer_content:
                    reward = 1.0
                else:
                    reward = 0.0
                    
            except Exception as e:
                print(f"Error processing answer: {e}")
                reward = 0.0  # Return 0 points when an error occurs

            rewards.append(reward)

        return rewards

orms['choice_accuracy'] = TestAccuracy

from swift.plugin.orm import orms
orms['choice_accuracy'] = TestAccuracy


if __name__ == "__main__":
    sample = """
    <think>
        The agent should first navigate to the apple, then pick it up.
    </think>
    <answer>
        [
            "navigate to Apple",
            "pick up Apple"
        ]
    </answer>
    """

    think_content, answer_content = TestAccuracy().extract_think_and_answer(sample)
    print("think_content:")
    print(think_content)
    print("\nanswer_content:")
    print(answer_content)