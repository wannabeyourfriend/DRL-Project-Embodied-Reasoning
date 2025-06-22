import os
import re
import warnings
import json
from typing import Dict, List, Union, Optional

from swift.plugin.orm import ORM

orms = {}

class DecisionFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<Thinking>.*?</Thinking>\s*<DecisionMaking>.*?</DecisionMaking>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


orms['decision_format'] = DecisionFormat

from swift.plugin.orm import orms
orms['decision_format'] = DecisionFormat