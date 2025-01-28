import os
import requests
from .base import LLMProvider

class PerplexityProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.environ.get("PERPLEXITY_API_KEY")
        self.api_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    @property
    def name(self) -> str:
        return "sonar-medium-chat"

    def generate_response(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            payload = {
                "model": "sonar",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": 1000,
                "top_p": 0.9,
                "stream": False,
                "response_format": {
                    "type": "json_object",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "model": {"type": "string"},
                            "object": {"type": "string"},
                            "created": {"type": "integer"},
                            "citations": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "choices": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "index": {"type": "integer"},
                                        "finish_reason": {"type": "string"},
                                        "message": {
                                            "type": "object",
                                            "properties": {
                                                "role": {"type": "string"},
                                                "content": {"type": "string"}
                                            }
                                        },
                                        "delta": {
                                            "type": "object",
                                            "properties": {
                                                "role": {"type": "string"},
                                                "content": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            },
                            "usage": {
                                "type": "object",
                                "properties": {
                                    "prompt_tokens": {"type": "integer"},
                                    "completion_tokens": {"type": "integer"},
                                    "total_tokens": {"type": "integer"}
                                }
                            }
                        }
                    }
                }
            }

            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )

            if response.status_code != 200:
                return f"Error with Perplexity API: {response.text}"

            data = response.json()
            content = data['choices'][0]['message']['content']

            # Include citations if they exist
            if 'citations' in data:
                citations = data['citations']
                content += "\n\nSources:\n" + "\n".join(f"- {citation}" for citation in citations)

            return content

        except Exception as e:
            return f"Error with Perplexity: {str(e)}"