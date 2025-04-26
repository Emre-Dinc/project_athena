import logging
from typing import List, Dict, Any
import numpy as np
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from core.config import config

logger = logging.getLogger(__name__)

class VectorDB:
    """Vector database handler for Project Athena using Milvus."""

    def __init__(self):
        self.host = config.get_system_config("milvus_host")
        self.port = config.get_system_config("milvus_port")
        self.collection_name = config.get_system_config("milvus_collection")
        self.dim = 384
        self.collection = None
        self._connect()
        self._init_collection()

    def _connect(self):
        try:
            connections.connect(alias="default", host=self.host, port=self.port)
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Milvus connection error: {e}")
            raise

    def _init_collection(self):
        try:
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                logger.info(f"Using existing Milvus collection: {self.collection_name}")
            else:
                fields = [
                    FieldSchema(name="paper_id", dtype=DataType.VARCHAR, max_length=256, is_primary=True, auto_id=False),
                    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
                    FieldSchema(name="authors", dtype=DataType.VARCHAR, max_length=1024),
                    FieldSchema(name="paper_url", dtype=DataType.VARCHAR, max_length=1024),
                    FieldSchema(name="pdf_path", dtype=DataType.VARCHAR, max_length=1024),
                    FieldSchema(name="gpt_summary", dtype=DataType.VARCHAR, max_length=32000),
                    FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=1024),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
                ]
                schema = CollectionSchema(fields, description="Research papers for Athena")
                self.collection = Collection(self.collection_name, schema)
                self.collection.create_index(
                    field_name="embedding",
                    index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
                )
                logger.info(f"Created new Milvus collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Collection init failed: {e}")
            raise

    def add_paper(self, paper: Dict[str, Any]):
        """
        Add a single paper to the database.
        Args:
            paper: Dictionary with fields.
        """
        try:
            required_fields = ["paper_id", "title", "authors", "paper_url", "pdf_path", "gpt_summary", "tags", "embedding"]
            for field in required_fields:
                if field not in paper:
                    raise ValueError(f"Missing required field '{field}' in paper data.")

            # Convert embedding to correct type
            if isinstance(paper["embedding"], list):
                paper["embedding"] = np.array(paper["embedding"], dtype=np.float32)

            data_to_insert = [
                [paper["paper_id"]],
                [paper["title"]],
                [", ".join(paper["authors"]) if isinstance(paper["authors"], list) else paper["authors"]],
                [paper["paper_url"]],
                [paper["pdf_path"]],
                [paper["gpt_summary"]],
                [", ".join(paper["tags"]) if isinstance(paper["tags"], list) else paper["tags"]],
                [paper["embedding"].tolist()]
            ]

            self.collection.insert(data_to_insert)
            logger.info(f"Inserted paper into Milvus: {paper['title']}")

        except Exception as e:
            logger.error(f"Failed to insert paper {paper.get('title', 'unknown')}: {e}")
            raise

    # Rest of search_papers_by_embedding, delete_paper_by_id, disconnect stay the same