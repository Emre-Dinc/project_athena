"""
Exa Scraper Rewritten for Project Athena
Full metadata + live crawl + PDF access + robust handling
"""

import logging
import requests
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from urllib.parse import urlparse
import json

from core.config import config

logger = logging.getLogger(__name__)
@dataclass
class SearchParams:
    query: str
    max_results: int = 10
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    domains: Optional[List[str]] = None
    live_crawl: str = "always"

class ExaScraper:
    def __init__(self):
        self.api_key = config.get_api_key("exa")
        if not self.api_key:
            raise ValueError("Exa API key missing")

        self.base_url = "https://api.exa.ai"
        self.search_endpoint = f"{self.base_url}/search"
        self.contents_endpoint = f"{self.base_url}/contents"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        self.timeout = 30
        self.page_size = 10

    def search(self, params: SearchParams, download_dir: Path) -> List[Dict[str, Any]]:
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Starting search for: {params.query}")
        results = []
        offset = 0

        while len(results) < params.max_results:
            batch = self._fetch_batch(params, offset)
            if not batch:
                break
            results.extend(batch)
            offset += len(batch)

        papers = []
        for result in results[:params.max_results]:
            processed = self._process_result(result)
            if processed:
                papers.append(processed)

        logger.info(f"Found {len(papers)} valid papers from Exa")
        return papers

    def _fetch_batch(self, params: SearchParams, offset: int) -> List[Dict[str, Any]]:
        payload = {
            "query": params.query,
            "numResults": min(self.page_size, params.max_results - offset),
            "offset": offset,
            "livecrawl": params.live_crawl,
            "includeDomains": params.domains or ["arxiv.org"],
            "startYear": params.start_year or 2000,
            "endYear": params.end_year or datetime.now().year,
        }

        try:
            res = requests.post(self.search_endpoint, headers=self.headers, json=payload, timeout=self.timeout)
            if res.status_code == 200:
                return res.json().get("results", [])
            logger.warning(f"Request failed with {res.status_code}, retrying...")
            time.sleep(2)
        except Exception as e:
            logger.error(f"Fetch batch failed: {e}")
        return []

    def _process_result(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            title = result.get("title")
            url = result.get("url")
            if not title or not url:
                return None

            exa_id = result.get("id")
            full_content = self._fetch_contents(exa_id) if exa_id else {}

            pdf_url = self._get_pdf_url(url)
            # Assign local_pdf_path based on url, do not download here
            if pdf_url:
                filename = "paper_" + urlparse(url).path.split("/")[-1].replace(".pdf", "") + ".pdf"
                local_pdf_path = self.download_dir / filename
            else:
                local_pdf_path = self.download_dir / "paper_.pdf"
            author_field = result.get("author", "")
            raw_splits = [a.strip() for a in author_field.split(",")]

            authors = []
            for item in raw_splits:
                if "@" in item:
                    continue  # Skip emails
                words = item.strip().split()
                if 1 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()):
                    authors.append(item.strip())
            processed = {
                "title": title.strip(),
                "url": url,
                "authors": authors,  # <- Here
                "abstract": result.get("extract", ""),
                "summary": result.get("summary", ""),
                "venue": result.get("publishedDate", ""),
                "year": full_content.get("metadata", {}).get("year"),
                "pdf_url": pdf_url,
                "pdf_path": str(local_pdf_path),
                "html_fallback_url": url if not local_pdf_path.exists() else "",
                "full_content": full_content.get("content", ""),
                "source": "exa"
            }
            logger.info(processed["authors"])
            return processed
        except Exception as e:
            logger.error(f"Error processing result: {e}")
            return None

    def _fetch_contents(self, doc_id: str) -> Dict[str, Any]:
        try:
            res = requests.post(self.contents_endpoint, headers=self.headers, json={"ids": [doc_id]}, timeout=self.timeout)
            if res.status_code == 200:
                return res.json().get("results", [{}])[0]
        except Exception as e:
            logger.error(f"Failed to fetch contents for {doc_id}: {e}")
        return {}

    def _get_pdf_url(self, url: str) -> Optional[str]:
        url = url.strip().lower()
        if url.endswith(".pdf"):
            return url
        if "arxiv.org" in url:
            if "/abs/" in url:
                paper_id = url.split("/abs/")[-1].split("?")[0]
                return f"https://arxiv.org/pdf/{paper_id}.pdf"
            if "/pdf/" in url:
                return url if url.endswith(".pdf") else url + ".pdf"
        return None

    def _download_pdf(self, pdf_url: str) -> Optional[Path]:
        try:
            filename = "paper_" + pdf_url.split("/")[-1]
            local_path = self.download_dir / filename
            if local_path.exists():
                return local_path

            logger.info(f"Downloading PDF: {pdf_url}")
            with requests.get(pdf_url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return local_path
        except Exception as e:
            logger.error(f"Failed to download PDF: {e}")
            return None
