from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, list_collections
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class ConceptVectorDB:
    def __init__(self, collection_name: str = "concept_vectors", dim: int = 384):
        self.collection_name = collection_name
        self.dim = dim
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        connections.connect()
        self._init_collection()

    def _init_collection(self):
        if self.collection_name in list_collections():
            self.collection = Collection(self.collection_name)
        else:
            fields = [
                FieldSchema(name="concept", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            ]
            schema = CollectionSchema(fields, description="Concept vector store")
            self.collection = Collection(name=self.collection_name, schema=schema)
            self.collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}})
            self.collection.load()

    def embed_concept(self, concept: str) -> List[float]:
        return self.model.encode(concept).tolist()

    def add_concepts(self, concepts: List[str]):
        to_insert = []
        existing = self.get_existing_concepts()
        for concept in concepts:
            if concept not in existing:
                emb = self.embed_concept(concept)
                to_insert.append((concept, emb))

        if to_insert:
            try:
                self.collection.insert(to_insert)
                logger.info(f"Inserted {len(to_insert)} new concepts.")
            except Exception as e:
                logger.warning(f"Failed to insert concepts into Milvus: {e}")

    def get_existing_concepts(self) -> List[str]:
        self.collection.load()
        results = self.collection.query(expr="", output_fields=["concept"], limit=9999)
        return [r.get("concept") for r in results]

    def search_similar_concepts(self, query_text: str, top_k: int = 5) -> List[str]:
        embedding = [self.embed_concept(query_text)]
        self.collection.load()
        results = self.collection.search(embedding, "embedding", param={"metric_type": "L2", "params": {"nprobe": 10}}, limit=top_k, output_fields=["concept"])
        return [hit.entity.get("concept") for hit in results[0]]