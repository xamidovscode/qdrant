from __future__ import annotations

from typing import Any, Dict, Optional, List
import re
import requests
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
    Top-K natijalarni olish + kontekst yig‘ish.
    """

    def __init__(
        self,
        qdrant_url: str,
        collection: str,
        embedder: OpenRouterEmbedder,
        text_key: str = "text",
    ):
        self.q = QdrantClient(url=qdrant_url)
        self.collection = collection
        self.embedder = embedder
        self.text_key = text_key

        self._phone_re = re.compile(r"\+?\d[\d\s\-\(\)]{7,}\d")

    def _extract_text(self, payload: Dict[str, Any]) -> str:
        v = payload.get(self.text_key)
        if isinstance(v, str) and v.strip():
            return v.strip()

        for k in ("clean_text", "text", "body", "answer", "question"):
            vv = payload.get(k)
            if isinstance(vv, str) and vv.strip():
                return vv.strip()
        return ""

    def _is_noise(self, text: str) -> bool:
        if not text:
            return True
        t = text.strip()
        if len(t) < 40:
            return True
        if self._phone_re.search(t):
            return True
        return False

    def ask_many(
        self,
        question: str,
        *,
        top_k: int = 12,
        score_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        vector = self.embedder.embed(question)

        res = self.q.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        items: List[Dict[str, Any]] = []
        for p in res.points or []:
            payload = p.payload or {}
            text = self._extract_text(payload)
            score = float(p.score) if p.score is not None else None

            if score_threshold is not None and score is not None and score < score_threshold:
                continue

            items.append(
                {
                    "id": p.id,
                    "score": score,
                    "text": text,
                    "payload": payload,
                }
            )

        return {"found": len(items) > 0, "matches": items}

    def answer_text(
        self,
        matches: List[Dict[str, Any]],
        *,
        max_chars: int = 1800,
        max_chunks: int = 6,
    ) -> str:
        """
        Top-K natijadan foydali bo‘laklarni yig‘ib kontekst qiladi.
        Noise bo‘laklarni tashlab ketadi, lekin hammasi noise bo‘lsa baribir qaytaradi.
        """
        if not matches:
            return ""

        # score bo‘yicha sort (katta -> kichik)
        matches = sorted(matches, key=lambda x: (x["score"] or 0.0), reverse=True)

        selected: List[str] = []
        total = 0

        # 1) avval noise bo‘lmaganlarini olamiz
        for m in matches:
            t = (m.get("text") or "").strip()
            if not t or self._is_noise(t):
                continue
            if t in selected:
                continue

            if total + len(t) > max_chars:
                remaining = max_chars - total
                if remaining > 120:
                    selected.append(t[:remaining])
                break

            selected.append(t)
            total += len(t)
            if len(selected) >= max_chunks:
                break

        # 2) agar hammasi noise bo‘lib qolsa, top1 ni qaytarib yuboramiz
        if not selected:
            t = (matches[0].get("text") or "").strip()
            return t[:max_chars]

        return "\n\n---\n\n".join(selected)


"""
FASTAPI endpoint misol:

from fastapi import APIRouter
from pydantic import BaseModel
from decouple import config

router = APIRouter()

API_KEY = config("OPENROUTER_API_KEY")
QDRANT_URL = config("QDRANT_URL", default="http://localhost:6333")

embedder = OpenRouterEmbedder(api_key=API_KEY, model="text-embedding-3-small")
searcher = QdrantSemanticSearch(
    qdrant_url=QDRANT_URL,
    collection="play_kb",
    embedder=embedder,
    text_key="text",
)

class QuestionResponse(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    status: str
    matches: list  # debug uchun (xohlasang olib tashla)

@router.post("/question/", response_model=AnswerResponse)
async def question_api(body: QuestionResponse):
    res = searcher.ask_many(body.question, top_k=12, score_threshold=None)
    context = searcher.answer_text(res["matches"], max_chars=1800, max_chunks=6)

    if not context:
        return {"answer": "Topilmadi.", "status": "ok", "matches": []}

    return {"answer": context, "status": "ok", "matches": res["matches"][:5]}
"""
