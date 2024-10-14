import dtlpy as dl
import requests
import logging
import json
import os

logger = logging.getLogger("AzureAI Adapter")


class ModelAdapter(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        self.adapter_defaults.upload_annotations = False
        self.api_key = os.environ.get("AZURE_API_KEY")
        if self.api_key is None:
            raise ValueError(f"Missing API key")

        self.url = self.configuration.get("endpoint-url", "")
        if not self.url:
            raise ValueError("You must provide the endpoint URL for the deployed model. "
                             "Add the URL to the model's configuration under 'endpoint-url'.")

    def embed(self, batch, **kwargs):
        embeddings = []
        for item in batch:
            if isinstance(item, str):
                self.adapter_defaults.upload_features = True
                text = item
            else:
                self.adapter_defaults.upload_features = False
                try:
                    prompt_item = dl.PromptItem.from_item(item)
                    is_hyde = item.metadata.get('prompt', dict()).get('is_hyde', False)
                    if is_hyde is True:
                        messages = prompt_item.to_messages(model_name=self.configuration.get('hyde_model_name'))[-1]
                        if messages['role'] == 'assistant':
                            text = messages['content'][-1]['text']
                        else:
                            raise ValueError(f'Only assistant messages are supported for hyde model')
                    else:
                        messages = prompt_item.to_messages(include_assistant=False)[-1]
                        text = messages['content'][-1]['text']

                except ValueError as e:
                    raise ValueError(f'Only mimetype text or prompt items are supported {e}')

            data = {
                "input": [text],
                "input_type": "query"
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            s = requests.Session()
            response = s.post(self.url, data=json.dumps(data), headers=headers)
            if not response.ok:
                raise ValueError(f'error:{response.status_code}, message: {response.text}')

            embedding = response.json().get("data")[0].get("embedding")
            logger.info(f'Extracted embeddings for text {item}: {embedding}')
            embeddings.append(embedding)

        return embeddings


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()

    dl.setenv('rc')
    model = dl.models.get(model_id='670501fa265e3487d8ae0516')
    model.configuration[
        "endpoint-url"] = "https://Cohere-embed-v3-multilingual-ovr.swedencentral.models.ai.azure.com/embeddings"
    model.configuration["featureSetName"] = "testing3"
    model.configuration["embeddings_size"] = 1024
    model.update()
    item = dl.items.get(item_id='670ba346597b3128a2664998')
    adapter = ModelAdapter(model)
    adapter.embed_items(items=[item])
