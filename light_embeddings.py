import os
import requests
from langchain_core.embeddings import Embeddings

class LightHFEmbeddings(Embeddings):
    """Calls the Hugging Face Inference Providers router over HTTP, so no torch,
       transformers, or sentence-transformers ever get loaded into this process."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_name}/pipeline/feature-extraction"
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN environment variable is not set. "
                "Set it in your Render service's Environment tab."
            )
        self.headers = {"Authorization": f"Bearer {token}"}

    def _call_api(self, texts):
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
        )
        response.raise_for_status()
        return response.json()

    def embed_documents(self, texts):
        return self._call_api(texts)

    def embed_query(self, text):
        return self._call_api([text])[0]