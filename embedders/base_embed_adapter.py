import requests
import dtlpy as dl
import logging
import json
import os

logger = logging.getLogger("AzureAI Adapter")


class ModelAdapter(dl.BaseModelAdapter):
    def __init__(self, model_entity: dl.Model):
        self.api_key = os.environ.get("AZURE_API_KEY", None)
        if self.api_key is None:
            raise ValueError(f"Missing `AZURE_API_KEY` env var")
        super().__init__(model_entity)

    def load(self, local_path, **kwargs):
        self.url = self.configuration.get("endpoint-url", "")
        batch_size = self.configuration.get("batch_size", 10)
        if batch_size > 2048:
            raise ValueError(f'Model configuration `batch_size` cannot be larger that 2048. Value: {batch_size}')
        if not self.url:
            raise ValueError("You must provide the endpoint URL for the deployed model. "
                             "Add the URL to the model's configuration under 'endpoint-url'.")

    def prepare_item_func(self, item: dl.Item):
        if 'text' not in item.mimetype:
            raise ValueError('Unsupported data type: {}. This embeddings ')
        buffer = item.download(save_locally=False)
        text = buffer.read().decode()
        return text

    def call_model_requests(self, texts):
        # Configure payload data sending to API endpoint
        data = {"input": texts}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        response = requests.post(self.url, data=json.dumps(data), headers=headers)
        if not response.ok:
            raise ValueError(f'Failed getting response. status code:{response.status_code}, error: {response.text}')
        vectors = response.json().get('data')

        return vectors

    def embed(self, batch, **kwargs):
        vectors = self.call_model_requests(batch)
        return vectors
