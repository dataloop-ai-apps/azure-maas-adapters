import requests
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
        self.url = self.configuration.get("endpoint-url", "")
        if not self.url:
            raise ValueError("You must provide the endpoint URL for the deployed model. "
                             "Add the URL to the model's configuration under 'endpoint-url'.")

    def prepare_item_func(self, item: dl.Item):
        if ('json' not in item.mimetype or
                item.metadata.get('system', dict()).get('shebang', dict()).get('dltype') != 'prompt'):
            raise ValueError('Only prompt items are supported')
        buffer = json.load(item.download(save_locally=False))
        return buffer

    @staticmethod
    def extract_content(line):
        try:
            if line.startswith("data: "):
                line = line[len("data: "):]
            json_data = json.loads(line)
            if "choices" in json_data and json_data["choices"]:
                choice = json_data["choices"][0]
                if "delta" in choice and "content" in choice["delta"]:
                    return choice["delta"]["content"]
        except json.JSONDecodeError:
            pass
        return ""

    @staticmethod
    def post_stream(url, data, headers, stream):
        s = requests.Session()
        ans = ""
        response = s.post(url, data=json.dumps(data), headers=headers, stream=stream)
        if stream:
            with response:  # To properly closed The response object after the block of code is executed
                if not response.ok:
                    raise ValueError(f'error:{response.status_code}, message: {response.text}')
                logger.info("Streaming the response")
                for line in response.iter_lines():
                    if line:
                        print(line)
                        line = line.decode('utf-8')
                        ans += ModelAdapter.extract_content(line)
        else:
            ans = response.json().get('choices')[0].get('message').get('content')
        return ans

    def call_model_requests(self, messages):
        # Configure payload data sending to API endpoint
        stream = self.model_entity.configuration.get('stream', True)
        data = {
            "messages": messages,
            "max_tokens": self.configuration.get('max_tokens', 1024),
            "temperature": self.configuration.get('temperature', 0.5),
            "top_p": self.model_entity.configuration.get('top_p', 0.7),
            "stream": stream
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        full_answer = self.post_stream(url=self.url, data=data, headers=headers, stream=stream)

        return full_answer

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

                full_answer = self.call_model_requests(messages=messages)

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
    dl.setenv('rc')
    azure_api_key_name = ''
    model = dl.models.get(model_id='')
    item = dl.items.get(item_id='')
    adapter = ModelAdapter(model, '')
    adapter.predict_items(items=[item])
