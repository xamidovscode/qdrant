from typing import Any, Dict, Optional
import requests
from decouple import config
from qdrant_client import QdrantClient


class OpenRouterEmbedder:
    BASE_URL = "https://openrouter.ai/api/v1/embeddings"

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", timeout: int = 60):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def embed(self, text: str) -> list[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": text}

        r = requests.post(self.BASE_URL, headers=headers, json=payload, timeout=self.timeout)
        if r.status_code != 200:
            raise Exception(f"OpenRouter error {r.status_code}: {r.text}")

        return r.json()["data"][0]["embedding"]


class QdrantSemanticSearch:
    """
    Vazifa:
      - user savol beradi
      - savolni embedding qiladi
      - Qdrant'dan eng o‘xshash 1 ta pointni topadi
      - o‘sha point payload ichidan text (yoki body) ni qaytaradi
    """

    def __init__(
        self,
        qdrant_url: str,
        collection: str,
        embedder: OpenRouterEmbedder,
        text_key: str = "text",  # payload ichida qaysi key'da matn saqlangan
    ):
        self.q = QdrantClient(url=qdrant_url)
        self.collection = collection
        self.embedder = embedder
        self.text_key = text_key

    def ask(self, question: str, *, score_threshold: Optional[float] = None):
        vector = self.embedder.embed(question)

        res = self.q.query_points(
            collection_name=self.collection,
            query=vector,
            limit=1,
            with_payload=True,
        )

        if not res.points:
            return {"found": False, "text": None, "score": None, "payload": None}

        top = res.points[0]
        score = float(top.score) if top.score is not None else None

        if score_threshold is not None and score is not None and score < score_threshold:
            return {"found": False, "text": None, "score": score, "payload": top.payload}

        payload = top.payload or {}
        text = payload.get(self.text_key) or payload.get("body") or payload.get("question")

        return {
            "found": True,
            "text": text,
            "score": score,
            "payload": payload,
        }


if __name__ == "__main__":
    API_KEY = config("OPENROUTER_API_KEY")
    QDRANT_URL = config("QDRANT_URL", default="http://qdrant:6333")

    embedder = OpenRouterEmbedder(api_key=API_KEY, model="text-embedding-3-small")
    searcher = QdrantSemanticSearch(
        qdrant_url=QDRANT_URL,
        collection="play_kb",
        embedder=embedder,
        text_key="text",
    )

    q = "Narxlar haqida ma'lumot bormi?"
    res = searcher.ask(q, score_threshold=None)

    print("FOUND:", res["found"])
    print("SCORE:", res["score"])
    print("TEXT:", res["text"])
