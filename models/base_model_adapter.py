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
        stream = self.configuration.get("stream", True)
        max_tokens = self.configuration.get('max_tokens', 1024)
        temperature = self.configuration.get('temperature', 0.5)
        top_p = self.model_entity.configuration.get('top_p', 0.7)
        data = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        s = requests.Session()
        response = s.post(self.url, data=json.dumps(data), headers=headers, stream=stream)
        if not response.ok:
            raise ValueError(f'error:{response.status_code}, message: {response.text}')

        if stream is True:
            with response:  # To properly closed The response object after the block of code is executed
                logger.info("Streaming the response")
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        yield self.extract_content(line) or ""
        else:
            yield response.json().get('choices')[0].get('message').get('content')

    def predict(self, batch, **kwargs):
        system_prompt = self.model_entity.configuration.get('system_prompt', "")
        add_metadata = self.configuration.get("add_metadata")
        model_name = self.model_entity.name

        for prompt_item in batch:
            _messages = prompt_item.to_messages(
                model_name=model_name)  # Get all messages including model annotations

            messages = self.reformat_messages(_messages)
            messages.insert(0, {"role": "system",
                                "content": system_prompt})

            nearest_items = prompt_item.prompts[-1].metadata.get('nearestItems', [])
            if len(nearest_items) > 0:
                context = prompt_item.build_context(nearest_items=nearest_items,
                                                    add_metadata=add_metadata)
                messages.append({"role": "assistant", "content": context})

            stream_response = self.post_stream(messages=messages)
            response = ""
            for chunk in stream_response:
                #  Build text that includes previous stream
                response += chunk
                prompt_item.add(message={"role": "assistant",
                                         "content": [{"mimetype": dl.PromptType.TEXT,
                                                      "value": response}]},
                                model_info={'name': model_name,
                                            'confidence': 1.0,
                                            'model_id': self.model_entity.id})

        return []

    @staticmethod
    def reformat_messages(messages):
        """
        Convert OpenAI message format to HTTP request format.
        This function takes messages formatted for OpenAI's API and transforms them into a format suitable for HTTP
        requests.

        :param messages: A list of messages in the OpenAI format.
        :return: A list of messages reformatted for HTTP requests.
        """
        reformat_messages = list()
        for message in messages:
            content = message["content"]
            question = content[0][content[0].get("type")]
            role = message["role"]

            reformat_message = {"role": role, "content": question}
            reformat_messages.append(reformat_message)

        return reformat_messages
