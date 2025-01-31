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
                "stream": False
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

            # Create comma-separated output with content and citations
            output_parts = [content]

            # Add citations if they exist
            if 'citations' in data:
                output_parts.extend(data['citations'])

            # Join all parts with commas
            return ','.join(output_parts)

        except Exception as e:
            return f"Error with Perplexity: {str(e)}"
