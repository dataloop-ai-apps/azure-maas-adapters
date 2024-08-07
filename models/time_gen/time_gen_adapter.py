from openai import OpenAI
import dtlpy as dl
import logging
import json
import os

logger = logging.getLogger("AzureAI Adapter")


class ModelAdapter(dl.BaseModelAdapter):
    def __init__(self, model_entity: dl.Model, azure_api_key_name):
        self.api_key = os.environ.get(azure_api_key_name, None)
        if self.api_key is None:
            raise ValueError(f"Missing API key: {azure_api_key_name}")
        super().__init__(model_entity)

    def load(self, local_path, **kwargs):
        url = self.configuration.get("endpoint-url", None)
        if url is None:
            raise ValueError("You must provide the endpoint URL for the deployed model. "
                             "Add the URL to the model's configuration under 'endpoint-url'.")

        self.client = OpenAI(base_url=url, api_key=self.api_key)

    def prepare_item_func(self, item: dl.Item):
        if ('json' not in item.mimetype or
                item.metadata.get('system', dict()).get('shebang', dict()).get('dltype') != 'prompt'):
            raise ValueError('Only prompt items are supported')
        buffer = json.load(item.download(save_locally=False))
        return buffer

    def predict(self, batch, **kwargs):
        system_prompt = self.model_entity.configuration.get('system_prompt', "")

        annotations = []
        for prompt_item in batch:
            collection = dl.AnnotationCollection()
            for prompt_name, prompt_content in prompt_item.get('prompts').items():
                # get latest question
                question = [p['value'] for p in prompt_content if 'text' in p['mimetype']][0]
                messages = [{"role": "system",
                             "content": system_prompt},
                            {"role": "user",
                             "content": question}]
                nearest_items = [p['nearestItems'] for p in prompt_content if 'metadata' in p['mimetype'] and
                                 'nearestItems' in p]
                if len(nearest_items) > 0:
                    nearest_items = nearest_items[0]
                    # build context
                    context = ""
                    for item_id in nearest_items:
                        context_item = dl.items.get(item_id=item_id)
                        with open(context_item.download(), 'r', encoding='utf-8') as f:
                            text = f.read()
                        context += f"\n{text}"
                    messages.append({"role": "assistant", "content": context})

                response = self.client.chat.completions.create(
                    model="azureai",
                    messages=messages,
                    temperature=self.configuration.get('temperature', 0.5),
                    top_p=self.configuration.get('top_p', 1),
                    max_tokens=self.configuration.get('max_tokens', 1024),
                    stream=True
                )
                full_answer = ""
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        full_answer += chunk.choices[0].delta.content
                collection.add(
                    annotation_definition=dl.FreeText(text=full_answer),
                    prompt_id=prompt_name,
                    model_info={
                        'name': self.model_entity.name,
                        'model_id': self.model_entity.id,
                        'confidence': 1.0
                    }
                )
            annotations.append(collection)
        return annotations


if __name__ == '__main__':
    azure_api_key_name = ''
    model = dl.models.get(model_id='')
    item = dl.items.get(item_id='')
    adapter = ModelAdapter(model, '')
    adapter.predict_items(items=[item])
