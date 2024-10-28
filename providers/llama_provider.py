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
        return "codellama-34b-instruct"

    def generate_response(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            response = self.client.chat.completions.create(
                model="codellama/CodeLlama-34b-Instruct-hf",
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
