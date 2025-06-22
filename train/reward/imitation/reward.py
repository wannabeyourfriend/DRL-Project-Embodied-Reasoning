import re
from typing import List, Any, Dict

try:
    from swift.plugin.orm import ORM, orms
except ImportError:
    class ORM:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, completions: List[str], **kwargs: Any) -> List[float]:
            raise NotImplementedError
    orms: Dict[str, ORM] = {}


class ComprehensiveDecisionRewardORM(ORM):
    def __init__(self,
                 action_block_format_reward: float = 0.5,
                 single_step_penalty: float = -0.25,
                 require_thinking_tag: bool = False,
                 thinking_tag_missing_penalty: float = -0.5,
                 strict_thinking_decision_format_bonus: float = 0.2
                ):
        """
        Initializes the comprehensive reward model.

        Args:
            action_block_format_reward (float): Reward if the decision-making part of the completion
                                                (either the single block in strict format or the whole
                                                completion if not strict) is well-formatted with only
                                                <DecisionMaking> tags.
            single_step_penalty (float): Penalty applied if more than one action is generated
                                         for a single-step ground truth task.
            require_thinking_tag (bool): Whether a <Thinking> tag is generally expected.
            thinking_tag_missing_penalty (float): Penalty if require_thinking_tag is True and
                                                  no <Thinking> tag is found (especially if strict format fails).
            strict_thinking_decision_format_bonus (float): Bonus reward if the completion strictly follows
                                                           the <Thinking>...</Thinking><DecisionMaking>...</DecisionMaking>
                                                           format.
        """
        super().__init__()
        self.action_block_format_reward = action_block_format_reward
        self.single_step_penalty = single_step_penalty
        self.require_thinking_tag = require_thinking_tag
        self.thinking_tag_missing_penalty = thinking_tag_missing_penalty
        self.strict_thinking_decision_format_bonus = strict_thinking_decision_format_bonus

        self.strict_td_pattern = re.compile(
            r'^<Thinking>.*?</Thinking>\s*<DecisionMaking>.*?</DecisionMaking>(?![\s\S])',
            re.DOTALL | re.MULTILINE
        )
        self.strict_dm_content_extractor = re.compile(
            r'^<Thinking>.*?</Thinking>\s*(<DecisionMaking>.*?</DecisionMaking>)(?![\s\S])',
            re.DOTALL | re.MULTILINE
        )
        self.dm_action_parser = re.compile(r"<DecisionMaking>(.*?)</DecisionMaking>")

    def _parse_actions_from_block(self, text_block: str) -> List[str]:
        """Extracts action names from a string containing <DecisionMaking>tags."""
        actions = self.dm_action_parser.findall(text_block)
        return [action.strip() for action in actions]

    def _check_action_block_integrity(self, action_block_str: str) -> bool:
        """
        Checks if the action_block_str strictly consists of <DecisionMaking> tags
        and whitespace, with no other text outside these tags.
        An empty string is considered well-formatted.
        """
        processed_str = self.dm_action_parser.sub("", action_block_str)
        return not processed_str.strip()

    def _calculate_r_nk(self, n: int, k: int) -> float:
        """Calculates the multi-step reward allocation R(n; k) = n(n+1) / k(k+1)."""
        if k == 0:  # Ground truth is an empty plan
            return 1.0 if n == 0 else 0.0  # Perfect match if model also predicts empty
        n_capped = min(n, k) # n cannot be greater than k by definition of prefix matching.
        return (n_capped * (n_capped + 1.0)) / (k * (k + 1.0))

    def __call__(self, completions: List[str], action: List[List[str]], **kwargs: Any) -> List[float]:
        """
        Calculates the total reward for a batch of completions.

        Args:
            completions (list[str]): List of model-generated strings.
            solution (list[list[str]]): List of ground truth action name sequences.
                                         Each item in the list is a sequence of actions for one sample.
        Returns:
            list[float]: Total reward for each completion.
        """
        batch_total_rewards = []
        solution = action  # Assuming 'action' is the ground truth action sequences
        if len(completions) != len(solution):
            raise ValueError("Completions and solutions lists must have the same length.")

        for completion_str, gold_action_sequence in zip(completions, solution):
            current_reward = 0.0
            dm_processing_block = completion_str  # Default: process the whole completion for DM tags

            # 1. Check for the strict <Thinking>...</Thinking><DecisionMaking>...</DecisionMaking> format
            strict_match = self.strict_td_pattern.match(completion_str)

            if strict_match:
                current_reward += self.strict_thinking_decision_format_bonus
                # Extract the single <DecisionMaking>...</DecisionMaking> block for focused processing
                dm_block_match = self.strict_dm_content_extractor.match(completion_str)
                if dm_block_match:
                    dm_processing_block = dm_block_match.group(1)
            else:
                # Strict format not met. Check for thinking tag presence if required.
                if self.require_thinking_tag:
                    if not re.search(r"<Thinking>.*?</Thinking>", completion_str, re.DOTALL):
                        current_reward += self.thinking_tag_missing_penalty

            # 2. Process the identified decision-making block (dm_processing_block)
            # 2a. Format Reward for the action block
            if self._check_action_block_integrity(dm_processing_block):
                current_reward += self.action_block_format_reward
            # else: Consider a penalty for bad action block format, or let accuracy handle it.

            # 2b. Accuracy Reward based on action sequence matching
            predicted_action_names = self._parse_actions_from_block(dm_processing_block)

            k = len(gold_action_sequence)  # Length of the ground truth action sequence
            n = 0  # Number of consecutively matched steps from the beginning
            for i in range(min(len(predicted_action_names), k)):
                if predicted_action_names[i] == gold_action_sequence[i]:
                    n += 1
                else:
                    break  # Stop matching once a mismatch is found

            r_accuracy_base = self._calculate_r_nk(n, k)

            # Apply penalty for single-step tasks if model generates more than one action
            accuracy_penalty = 0.0
            if k == 1 and len(predicted_action_names) > 1:
                accuracy_penalty = self.single_step_penalty

            r_accuracy = r_accuracy_base + accuracy_penalty
            current_reward += r_accuracy

            batch_total_rewards.append(current_reward)

        return batch_total_rewards

# How to register this merged ORM (example):
# This would typically be in your plugin.py or a similar setup file.
# Ensure `orms` is the correct registry dictionary from your SWIFT environment.

orms['comprehensive_decision_reward'] = ComprehensiveDecisionRewardORM
#     action_block_format_reward=0.4,         # Tune this
#     single_step_penalty=-0.2,               # Tune this
#     require_thinking_tag=True,              # Tune this
#     thinking_tag_missing_penalty=-0.1,      # Tune this
#     strict_thinking_decision_format_bonus=0.3 # Tune this
# )

# To use it in your SWIFT training command, you would update:
# --reward_funcs 'comprehensive_decision_reward'
# And ensure --external_plugins points to the file containing this class and registration.