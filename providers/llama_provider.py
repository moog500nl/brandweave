import os
from openai import OpenAI
from .base import LLMProvider

class LlamaProvider(LLMProvider):
    def __init__(self):
        self.client = OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.environ.get("FIREWORKS_API_KEY")
        )
        
    @property
    def name(self) -> str:
        return "llama-v3p1-70b-instruct"

    def generate_response(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model="accounts/fireworks/models/llama-v3p1-70b-instruct",
                messages=messages,
                temperature=temperature
            )
            
            content = response.choices[0].message.content
            if content is None:
                return "Error: Llama returned empty response"
            return content
            
        except Exception as e:
            return f"Error with Llama: {str(e)}"
