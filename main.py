from __future__ import annotations

from typing import Any, Dict, List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


QDRANT_URL = "http://localhost:6333"
COLLECTION = "play_kb"
VECTOR_SIZE = 4


def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION):
        return

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


def upsert_points(client: QdrantClient, items: List[Dict[str, Any]]) -> None:
    points = [
        PointStruct(
            id=item["id"],
            vector=item["vector"],
            payload=item.get("payload", {}),
        )
        for item in items
    ]

    client.upsert(collection_name=COLLECTION, points=points)


def show_latest(client: QdrantClient, limit: int = 10) -> None:
    res = client.scroll(
        collection_name=COLLECTION,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    points, _next = res
    for p in points:
        print(f"id={p.id} payload={p.payload}")


def main() -> None:
    client = QdrantClient(url=QDRANT_URL)

    ensure_collection(client)

    data = [
        {
            "id": 1,
            "vector": [1.0, 0.0, 0.0, 0.0],
            "payload": {"question": "kpi nima?", "answer": "KPI — samaradorlik ko‘rsatkichlari"},
        },
        {
            "id": 2,
            "vector": [0.9, 0.1, 0.0, 0.0],
            "payload": {"question": "kpi degani nima", "answer": "KPI — performance ko‘rsatkichlari"},
        },
        {
            "id": 3,
            "vector": [0.0, 1.0, 0.0, 0.0],
            "payload": {"question": "pricing nima", "answer": "Pricing — narxlash jarayoni"},
        },
    ]

    upsert_points(client, data)
    print("✅ Qo'shildi. Hozir bazadagi pointlar:")
    show_latest(client, limit=20)


if __name__ == "__main__":
    main()
