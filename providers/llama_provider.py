import os
from openai import OpenAI
from .base import LLMProvider

class LlamaProvider(LLMProvider):
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get("AIMLAPI_KEY"),
            base_url="https://api.aimlapi.com/v1"
        )
        
    @property
    def name(self) -> str:
        return "llama-3.1-70b-instruct-turbo"

    def generate_response(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            response = self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=256
            )
            content = response.choices[0].message.content
            if content is None:
                return "Error: Llama returned empty response"
            return content
        except Exception as e:
            return f"Error with Llama: {str(e)}"
