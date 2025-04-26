"""
Rewritten Obsidian Injector Module for Project Athena.
Handles integration with Obsidian for structured knowledge management.
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml
from datetime import datetime
from unicodedata import normalize
from core.config import config
from modules.concept_extractor import ConceptExtractor
from modules.concept_vector_db import ConceptVectorDB

logger = logging.getLogger(__name__)

class ObsidianInjector:
    def __init__(self):
        self.vault_path = Path(config.get_system_config("obsidian_vault_path"))
        self.cache_dir = Path(config.DATA_DIR) / "obsidian_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.vault_path.exists():
            raise ValueError("Obsidian vault path does not exist. Configure it correctly.")

        logger.info(f"Obsidian Injector initialized. Vault: {self.vault_path}")

    def inject_papers(self, papers: List[Dict[str, Any]]) -> None:
        for paper in papers:
            try:
                self.inject_single_paper(paper)
            except Exception as e:
                logger.error(f"Error injecting paper '{paper.get('title', 'Unknown')}': {str(e)}")

    def push_single_paper(self, paper: Dict[str, Any], query_name: Optional[str] = None) -> None:
        if query_name:
            paper["query_name"] = query_name
        self.inject_single_paper(paper)

    def inject_single_paper(self, paper: Dict[str, Any]) -> None:
        paper = self._sanitize_paper(paper)
        note_path = self._build_note_path(paper)
        content = self._build_note_content(paper)
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(content, encoding="utf-8")
        self._cache_paper(paper)
        logger.info(f"Successfully injected: {note_path.relative_to(self.vault_path)}")

    def _sanitize_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        defaults = {
            "title": "Untitled Paper",
            "authors": [],
            "year": "",
            "venue": "",
            "url": "",
            "semantic_tags": [],
            "gpt_summary": "No summary available.",
            "refined_insights": "",
            "query_name": ""
        }
        for key, value in defaults.items():
            paper.setdefault(key, value)
        return paper

    def _slugify(self, text: str) -> str:
        text = normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        text = text.lower().strip()
        text = re.sub(r'[^\w\s-]', '', text)
        return re.sub(r'[-\s]+', '-', text)

    def _build_note_path(self, paper: Dict[str, Any]) -> Path:
        query_slug = self._slugify(paper["query_name"]) if paper.get("query_name") else (self._slugify(paper["semantic_tags"][0]) if paper.get("semantic_tags") else "general")
        query_folder = self.vault_path / query_slug
        primary_tag = self._slugify(paper["semantic_tags"][0]) if paper["semantic_tags"] else "general"
        full_folder = query_folder / primary_tag
        filename = self._slugify(paper["title"]) + ".md"
        return full_folder / filename

    def _build_note_content(self, paper: Dict[str, Any]) -> str:
        frontmatter = {
            "title": paper["title"],
            "authors": paper["authors"],
            "year": paper["year"],
            "venue": paper["venue"],
            "url": paper["url"],
            "tags": [self._slugify(tag) for tag in paper["semantic_tags"]],
            "date": datetime.now().strftime("%Y-%m-%d"),
            "type": "research-paper",
            "status": "unread"
        }
        sections = [
            "---",
            yaml.dump(frontmatter, sort_keys=False, allow_unicode=True).strip(),
            "---",
            "",
            "# 📚 Research Paper Notes",
            "",
            "## 📝 Summary",
            "",
            self._add_internal_links(paper["gpt_summary"], exclude=paper["title"]),
            "",
            "## 💡 Key Insights",
            "",
            self._add_internal_links(paper["refined_insights"], exclude=paper["title"]),
            "",
            "## 🔁 Concepts Mentioned",
            "",
            *self._build_concept_backlinks(paper["gpt_summary"]),
            "",
            "## 📋 Tasks",
            "",
            "- [ ] Read paper",
            "- [ ] Take detailed notes",
            "- [ ] Summarize key findings",
            "- [ ] Identify potential applications",
        ]
        return "\n".join(sections)

    def _build_concept_backlinks(self, summary: str) -> List[str]:
        concepts = ConceptExtractor.extract_concepts_from_summary(summary)
        backlinks = []
        db = ConceptVectorDB()
        for concept in concepts:
            matches = db.search_similar_concepts(concept)
            if matches:
                backlinks.append(f"- [[{concept}]] — mentioned in {len(matches)} notes")
        return backlinks or ["No major concepts extracted."]

    def _add_internal_links(self, text: str, exclude: str = "") -> str:
        # Minimal linking logic (can be expanded)
        text = text or ""
        titles_to_link = {self._slugify(exclude)}
        pattern = re.compile(r'\b(' + '|'.join(map(re.escape, titles_to_link)) + r')\b', re.IGNORECASE)
        return pattern.sub(lambda m: f"[[{m.group(0)}]]", text)

    def _cache_paper(self, paper: Dict[str, Any]) -> None:
        cache_id = self._slugify(paper["title"])
        cache_file = self.cache_dir / f"{cache_id}.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"title": paper["title"], "timestamp": datetime.now().isoformat()}, f)

# Usage example (manual run)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    injector = ObsidianInjector()
    test_paper = {
        "title": "A Theory of Swarm Intelligence",
        "authors": ["Jane Smith", "John Doe"],
        "url": "https://arxiv.org/abs/9999.99999",
        "year": "2024",
        "venue": "ICML",
        "semantic_tags": ["swarm intelligence", "decentralized systems"],
        "gpt_summary": "This paper proposes a unified model for decentralized problem solving based on swarm intelligence.",
        "refined_insights": "Insights into agent-based coordination and emergent behavior."
    }
    injector.inject_single_paper(test_paper)
