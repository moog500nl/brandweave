import os
from openai import OpenAI
from .base import LLMProvider

class DeepseekProvider(LLMProvider):
    def __init__(self):
        self.client = OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.environ.get("FIREWORKS_API_KEY")
        )
        
    @property
    def name(self) -> str:
        return "deepseek-v3"

    def generate_response(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model="accounts/fireworks/models/deepseek-v3",
                messages=messages,
                temperature=temperature,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            if content is None:
                return "Error: Deepseek returned empty response"
            return content
            
        except Exception as e:
            return f"Error with Deepseek: {str(e)}"
