import requests
import fitz  # PyMuPDF
import pdfplumber
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class PDFExtractor:
    def __init__(self):
        pass

    def _sanitize_filename(self, name: str) -> str:
        return "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in name).strip().replace(" ", "_")

    def download_pdf(self, url: str, save_path: Path) -> Optional[Path]:
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()

            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Successfully downloaded PDF to {save_path}")
            return save_path

        except Exception as e:
            logger.warning(f"Failed to download PDF from {url}: {e}")
            return None

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        try:
            with fitz.open(pdf_path) as doc:
                text = "\n".join(page.get_text() for page in doc)
            if len(text.strip()) > 100:
                return text
        except Exception as e:
            logger.warning(f"PyMuPDF extraction error: {e}")

        try:
            with pdfplumber.open(pdf_path) as doc:
                text = "\n".join(page.extract_text() or '' for page in doc.pages)
            return text
        except Exception as e:
            logger.warning(f"pdfplumber extraction error: {e}")
        return ""

    def extract_from_url(self, url: str, output_dir: Path, title: Optional[str] = None, paper_id: Optional[str] = None, output_path: Optional[Path] = None) -> dict:
        if output_path:
            pdf_path = output_path
        else:
            filename_base = self._sanitize_filename(paper_id or title or Path(urlparse(url).path).stem or "paper")
            pdf_path = output_dir / f"{filename_base}.pdf"

        logger.info(f"Downloading PDF from {url} to {pdf_path}")
        local_pdf = self.download_pdf(url, pdf_path)
        if not local_pdf or not local_pdf.exists():
            logger.error(f"PDF download failed or missing: {pdf_path}")
            return {
                "raw_text": "",
                "pdf_path": ""
            }
        logger.info(f"Extracting text from PDF file: {pdf_path}")
        raw_text = self.extract_text_from_pdf(local_pdf)
        return {
            "raw_text": raw_text,
            "pdf_path": str(local_pdf)
        }
