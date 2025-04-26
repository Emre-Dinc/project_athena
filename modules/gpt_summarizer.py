import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import hashlib
import re
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from core.config import config

logger = logging.getLogger(__name__)

class GPTSummarizer:
    def __init__(self):
        self.openai_api_key = config.get_api_key("openai")
        self.client = OpenAI(api_key=self.openai_api_key)
        self.cache_dir = Path(config.DATA_DIR) / "summaries_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("GPTSummarizer initialized.")

    def summarize(self, text: str) -> Dict[str, Any]:
        cache_key = self._get_cache_key(text, "summary_tags")
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return self._parse_gpt_output(cached_result)

        prompt = self._build_prompt(text)

        try:
            response = self.client.chat.completions.create(
                model=config.get_system_config("openai_model"),
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research analyst. Always return valid JSON with 'tags' and 'summary'."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.get_system_config("openai_max_tokens"),
                temperature=config.get_system_config("openai_temperature")
            )
            result = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating GPT summary: {str(e)}")
            return {"summary": "Error generating summary.", "tags": []}

        if not result:
            logger.warning("GPT response was empty.")
            return {"summary": "No summary generated.", "tags": []}

        parsed_result = self._parse_gpt_output(result)
        self._save_to_cache(cache_key, json.dumps(parsed_result, ensure_ascii=False))
        return parsed_result

    def get_embedding(self, text: str) -> List[float]:
        try:
            return self.embedding_model.encode(text, normalize_embeddings=True).tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return []

    def _parse_gpt_output(self, result: str) -> Dict[str, Any]:
        try:
            # Check if the result is wrongly double-quoted
            if result.startswith('"') and result.endswith('"'):
                logger.warning("Detected double-quoted GPT output, unescaping...")
                result = bytes(result[1:-1], "utf-8").decode("unicode_escape")

            match = re.search(r'\{.*\}', result, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
            else:
                raise ValueError("No JSON object found in result.")

            tags = parsed.get("tags", [])
            summary = parsed.get("gpt_summary", "") or parsed.get("summary", "")

            if not isinstance(tags, list):
                tags = []
            if not isinstance(summary, str):
                summary = ""

            if not summary.strip():
                logger.warning("Summary extracted but is empty.")

            summary = summary.replace("\n", "\n\n").strip()
            summary = re.sub(r'\n{3,}', '\n\n', summary)

            # Auto-fix if embedded JSON detected inside summary
            if tags == [] and summary.strip().startswith("{") and '"tags"' in summary:
                logger.warning("Detected embedded JSON inside summary. Auto-fixing...")
                try:
                    embedded = json.loads(summary)
                    tags = embedded.get("tags", [])
                    summary = embedded.get("summary", "")
                    summary = summary.replace("\n", "\n\n").strip()
                except Exception as e:
                    logger.error(f"Failed to auto-fix embedded JSON inside summary: {e}")

            return {"gpt_summary": summary, "tags": tags}

        except Exception as e:
            logger.warning(f"Failed to parse GPT JSON: {e}")
            logger.warning(f"Raw result: {repr(result)}")
            return {"gpt_summary": result.strip(), "tags": []}


    def _build_prompt(self, text: str) -> str:
        return f"""
You are a domain-general research analyst specializing in constructing a hyperlinked mathematical knowledge base.

Below is the raw text of a research paper.

Your tasks:

1. **Semantic Tag Extraction**  
   - Identify 12–18 essential keywords representing core mathematical models, algorithms, and theories.

2. **Analytical Deep-Dive Summary**  
   Produce a detailed Markdown-only summary with this strict structure:
   - ### Core Mathematical Problem
   - ### Theoretical Approach
   - ### Formal Mathematical Deep Dive
   - ### System Architecture and Logical Implementation
   - ### Critical Theoretical Limitations
   - ### Personal Analytical Reflection

Each section must:
- Include all key mathematical formulas, derivations, proofs, and assumptions.
- Render formulas in LaTeX using $$...$$ blocks.
- Explain every mathematical symbol and term.
- Deep dive into system architecture if any (include pseudocode if possible).
- Link important algorithms or concepts with double brackets: [[Deep Q-Learning]], [[Transformers]], etc.

3. **Output Format Rules**
- You must output a **single flat JSON object** with exactly two keys:
  - "tags" → List of strings.
  - "summary" → Single Markdown string.
- **NO** triple backticks, **NO** nested JSON inside summary.
- The summary string must be plain Markdown text **ready for direct insertion** into an Obsidian vault.
- The JSON must be fully parseable without extra escaping.
- No external commentary, no explanations, only the JSON.

4. **Strict Formatting Examples**
✅ GOOD:
{{
  "tags": ["chaotic systems", "genetic programming", "nonlinear ODEs"],
  "summary": "### Core Mathematical Problem\\nThe paper studies chaotic systems..."
}}
❌ BAD:
- Wrapping in ```json
- Double escaping
- Embedding JSON inside a Markdown block

---

**PAPER CONTENT:**
{text}
"""

    def _get_cache_key(self, text: str, operation: str) -> str:
        return f"{operation}_{hashlib.md5(text.encode('utf-8')).hexdigest()}.txt"

    def _check_cache(self, cache_key: str) -> Optional[str]:
        path = self.cache_dir / cache_key
        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        return None

    def _save_to_cache(self, cache_key: str, content: str) -> None:
        path = self.cache_dir / cache_key
        try:
            path.write_text(content, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
