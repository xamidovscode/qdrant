import requests


class OpenRouterEmbedder:
    BASE_URL = "https://openrouter.ai/api/v1/embeddings"

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model

    def embed(self, text: str) -> list:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "input": text,
        }

        response = requests.post(self.BASE_URL, headers=headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"Error: {response.text}")

        data = response.json()
        embedding = data["data"][0]["embedding"]

        return embedding


if __name__ == "__main__":
    API_KEY = "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"

    text = "Narxlar haqida ma'lumot bermoqchiman"

    embedder = OpenRouterEmbedder(api_key=API_KEY)
    vector = embedder.embed(text)

    print("Embedding uzunligi:", len(vector))
    print("Embedding vektor:")
    # print(vector)
