

import re
from typing import List


class ConceptExtractor:
    @staticmethod
    def extract_concepts_from_summary(summary: str) -> List[str]:
        """
        Extracts all [[concept]] mentions from a summary string.
        Example: "The model extends [[DQN]] and [[Transformer]]" -> ["DQN", "Transformer"]
        """
        return list(set(re.findall(r"\[\[([^\]]+)\]\]", summary)))