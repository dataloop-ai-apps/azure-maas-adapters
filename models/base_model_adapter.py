import dtlpy as dl
import requests
import logging
import json
import os

logger = logging.getLogger("AzureAI Adapter")


class ModelAdapter(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        self.adapter_defaults.upload_annotations = False
        self.adapter_defaults.clean_annotations = self.configuration.get("clean_annotations",
                                                                         True)  # TODO: add it to configuration?

        self.api_key = os.environ.get("AZURE_MODEL_API_KEY")
        if self.api_key is None:
            raise ValueError(f"Missing API key")

        self.url = self.configuration.get("endpoint-url", "")
        if not self.url:
            raise ValueError("You must provide the endpoint URL for the deployed model. "
                             "Add the URL to the model's configuration under 'endpoint-url'.")

    def prepare_item_func(self, item: dl.Item):
        prompt_item = dl.PromptItem.from_item(item)
        return prompt_item

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

    def post_stream(self, messages):

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

        s = requests.Session()
        response = s.post(self.url, data=json.dumps(data), headers=headers, stream=stream)
        if stream:
            try:
                with response:  # To properly closed The response object after the block of code is executed
                    if not response.ok:
                        raise ValueError(f'error:{response.status_code}, message: {response.text}')
                    logger.info("Streaming the response")
                    for line in response.iter_lines():
                        if line:
                            print(line)
                            line = line.decode('utf-8')
                            yield ModelAdapter.extract_content(line) or ""
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed: {e}")
        else:
            yield response.json().get('choices')[0].get('message').get('content')

    def predict(self, batch, **kwargs):
        system_prompt = self.model_entity.configuration.get('system_prompt', "")
        for prompt_item in batch:
            _messages = prompt_item.to_messages(
                model_name=self.model_entity.name)  # Get all messages including model annotations

            # REFORMAT FOR REQUESTS
            messages = list()
            for _message in _messages:
                content = _message["content"]
                question = content[0][content[0].get("type")]
                role = _message["role"]

                message = {"role": role, "content": question}
                messages.append(message)

            messages.insert(0, {"role": "system",
                                "content": system_prompt})

            nearest_items = prompt_item.prompts[-1].metadata.get('nearestItems', [])
            if len(nearest_items) > 0:
                context = prompt_item.build_context(nearest_items=nearest_items,
                                                    add_metadata=['system.document.source'])
                messages.append({"role": "assistant", "content": context})

            stream = self.post_stream(messages=messages)
            response = ""
            for chunk in stream:
                #  Build text that includes previous stream
                response += chunk
                prompt_item.add(message={"role": "assistant",
                                         "content": [{"mimetype": dl.PromptType.TEXT,
                                                      "value": response}]},
                                stream=True,
                                model_info={'name': self.model_entity.name,
                                            'confidence': 1.0,
                                            'model_id': self.model_entity.id})

        return []


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    dl.setenv('prod')
    model = dl.models.get(model_id='66af7a003824c67e50b125f1')
    item = dl.items.get(item_id='66b369a82e90de89dde976e0')
    adapter = ModelAdapter(model)
    adapter.predict_items(items=[item])
