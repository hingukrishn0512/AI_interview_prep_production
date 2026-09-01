import os
import requests
from langchain_core.embeddings import Embeddings


class LightHFEmbeddings(Embeddings):
    """Calls the Hugging Face Inference API directly over HTTP, so no torch,
       transformers, or sentence-transformers ever get loaded into this process.
       This keeps memory usage low enough to fit Render's free 512MB limit."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}

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