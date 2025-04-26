"""
Configuration module for Project Athena.
Handles loading environment variables, API keys, and system settings.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Define base project paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
LOGS_DIR = DATA_DIR / "logs"
CACHE_DIR = PROJECT_ROOT / ".cache"

# Ensure directories exist
PAPERS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env file
load_dotenv(PROJECT_ROOT / ".env")


class Config:
    """Configuration handler for Project Athena."""

    def __init__(self):
        """Initialize the configuration handler."""
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        self.api_keys = self._load_api_keys()
        self.system_config = self._load_system_config()
        self.logger.info("Configuration loaded successfully")

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_level = os.getenv("LOG_LEVEL", "INFO")
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)

        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(LOGS_DIR / "athena.log"),
                logging.StreamHandler()
            ]
        )

    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables."""
        keys = {
            "exa": os.getenv("EXA_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
        }

        # Log which keys are available (without revealing the actual keys)
        for key_name, key_value in keys.items():
            if key_value:
                self.logger.debug(f"{key_name.upper()} API key loaded")
            else:
                self.logger.warning(f"{key_name.upper()} API key not found in environment variables")

        return keys

    def _load_system_config(self) -> Dict[str, Any]:
        """Load system configuration from environment variables."""
        config = {
            "vector_db_type": os.getenv("VECTOR_DB_TYPE", "local"),
            "milvus_host": os.getenv("MILVUS_HOST", "localhost"),
            "milvus_port": int(os.getenv("MILVUS_PORT", "19530")),
            "milvus_collection": os.getenv("MILVUS_COLLECTION", "project_athena"),
            "obsidian_vault_path": os.getenv("OBSIDIAN_VAULT_PATH"),
            "batch_size": int(os.getenv("BATCH_SIZE", "10")),
            "openai_model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            "openai_temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
            "openai_max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "32000")),
        }

        return config

    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for a specific service.

        Args:
            service: The service name (exa, openai, anthropic)

        Returns:
            The API key or None if not found
        """
        return self.api_keys.get(service.lower())

    def get_system_config(self, key: str) -> Any:
        """Get system configuration value.

        Args:
            key: The configuration key

        Returns:
            The configuration value or None if not found
        """
        return self.system_config.get(key)

    @property
    def DATA_DIR(self):
        return DATA_DIR

    @property
    def PROJECT_ROOT(self):
        return PROJECT_ROOT

    @property
    def CACHE_DIR(self):
        return CACHE_DIR


# Create a singleton instance for global use
config = Config()
