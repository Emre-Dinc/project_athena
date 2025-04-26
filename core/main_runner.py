"""
Project Athena - Full Rewritten Main Runner
Handles orchestrating the cognition pipeline:
- Search papers
- Extract content
- Summarize + Tag
- Store vectors
- Inject into Obsidian
"""

import os
import sys
import signal
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from core.config import config
from modules.exa_scraper import SearchParams

logger = logging.getLogger(__name__)

# --- Data Directory ---
DATA_DIR = Path(config.PROJECT_ROOT) / "data"
sys.path.append(str(Path(__file__).parent.parent))

# --- State Tracker ---
@dataclass
class PipelineState:
    query: str
    total_papers: int
    processed: List[Dict[str, Any]]
    failed: List[Tuple[Dict[str, Any], str]]
    start_time: datetime
    last_save: datetime

    @property
    def success_rate(self) -> float:
        attempts = len(self.processed) + len(self.failed)
        return len(self.processed) / attempts if attempts else 0.0

    def to_json(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "total_papers": self.total_papers,
            "processed": self.processed,
            "failed": [{"paper": p, "error": e} for p, e in self.failed],
            "start_time": self.start_time.isoformat(),
            "last_save": self.last_save.isoformat()
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "PipelineState":
        return cls(
            query=data["query"],
            total_papers=data["total_papers"],
            processed=data["processed"],
            failed=[(item["paper"], item["error"]) for item in data["failed"]],
            start_time=datetime.fromisoformat(data["start_time"]),
            last_save=datetime.fromisoformat(data["last_save"])
        )

# --- Athena Orchestrator ---
class AthenaRunner:
    def __init__(self):
        logger.info("Initializing Athena Runner")
        self.config = config
        self._modules = {}
        self.state: Optional[PipelineState] = None
        self.state_file: Optional[Path] = None
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _get_module(self, name: str):
        if name not in self._modules:
            cls_map = {
                "exa_scraper": "ExaScraper",
                "pdf_extractor": "PDFExtractor",
                "gpt_summarizer": "GPTSummarizer",
                "vector_db": "VectorDB",
                "obsidian_injector": "ObsidianInjector"
            }
            module = __import__(f"modules.{name}", fromlist=[cls_map[name]])
            self._modules[name] = getattr(module, cls_map[name])()
        return self._modules[name]

    def _sanitize(self, text: str) -> str:
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text).strip("_").replace(" ", "_")

    def run_pipeline(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        workspace = DATA_DIR / self._sanitize(query)
        (workspace / "papers").mkdir(parents=True, exist_ok=True)
        (workspace / "raw_texts").mkdir(exist_ok=True)
        (workspace / "summaries").mkdir(exist_ok=True)
        (workspace / "concept_vectors").mkdir(exist_ok=True)
        (workspace / ".cache").mkdir(exist_ok=True)

        self.state_file = workspace / ".cache" / f"pipeline_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.state = PipelineState(query, 0, [], [], datetime.now(), datetime.now())

        exa_scraper = self._get_module("exa_scraper")
        papers = exa_scraper.search(SearchParams(query=query, max_results=max_results), download_dir=workspace / "papers")
        if not papers:
            logger.error("No papers found.")
            return []

        self.state.total_papers = len(papers)

        for idx, paper in enumerate(papers):
            try:
                logger.info(f"Processing paper {idx + 1}/{len(papers)}: {paper.get('title', 'Unknown')}")
                self._process_paper(paper, workspace)
            except Exception as e:
                logger.error(f"Failed to process paper: {e}")
                self.state.failed.append((paper, str(e)))
            finally:
                self._auto_save(workspace)

        logger.info(f"Pipeline finished. Success rate: {self.state.success_rate:.2%}")
        return self.state.processed

    def _process_paper(self, paper: Dict[str, Any], workspace: Path):
        pdf_extractor = self._get_module("pdf_extractor")
        gpt_summarizer = self._get_module("gpt_summarizer")
        vector_db = self._get_module("vector_db")
        obsidian_injector = self._get_module("obsidian_injector")

        # --- Step 1: Extract text from PDF ---
        text_data = pdf_extractor.extract_from_url(paper.get("pdf_url"), output_dir=workspace / "raw_texts")

        raw_text = text_data.get("raw_text", "")
        if not raw_text or len(raw_text) < 300:
            raise ValueError("Extracted content too short.")

        # Always ensure pdf_path is a string
        pdf_path = text_data.get("pdf_path", "")
        if isinstance(pdf_path, bytes):
            pdf_path = pdf_path.decode("utf-8")
        paper["pdf_path"] = pdf_path

        # --- Step 2: Generate fallback paper ID if missing ---
        if not paper.get("id"):
            paper["id"] = str(hash(paper.get("title", "")))

        # --- Step 3: Summarize and Tag ---
        summary_data = gpt_summarizer.summarize(raw_text)
        paper["gpt_summary"] = summary_data.get("gpt_summary", "")
        paper["semantic_tags"] = summary_data.get("tags", [])

        # Optionally overwrite 'summary' to match 'gpt_summary'
        paper["summary"] = paper["gpt_summary"]

        # --- Step 4: Get Embedding ---
        paper["embedding"] = gpt_summarizer.get_embedding(raw_text)


        # --- Step 5: Store Paper into Vector DB ---
        vector_db.add_paper({
            "paper_id": paper.get("id", ""),
            "title": paper.get("title", ""),
            "authors": paper.get("authors", []),
            "paper_url": paper.get("url", ""),
            "pdf_path": paper.get("pdf_path", ""),
            "gpt_summary": paper.get("gpt_summary", ""),
            "tags": paper.get("semantic_tags", []),
            "embedding": paper.get("embedding", [])
        })

        # --- Step 6: Push Paper to Obsidian ---
        obsidian_injector.push_single_paper(paper, query_name=self.state.query)
        # --- Step 7: Mark as processed ---
        self.state.processed.append(paper)

    def _auto_save(self, workspace: Path):
        if self.state and self.state_file:
            self.state.last_save = datetime.now()
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state.to_json(), f, indent=2)
            logger.info(f"Auto-saved pipeline state to {self.state_file}")

    def _handle_interrupt(self, signum, frame):
        logger.warning("Interrupt received. Saving state and exiting...")
        self._auto_save(DATA_DIR)
        sys.exit(0)

# --- Main Entrypoint ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    runner = AthenaRunner()
    runner.run_pipeline(query="Chaos genetic programming implementation and theory", max_results=5)