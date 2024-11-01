import os
from openai import OpenAI
from .base import LLMProvider

class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    @property
    def name(self) -> str:
        return "chatgpt-4o-latest"

    def generate_response(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            response = self.client.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=1000
            )
            content = response.choices[0].message.content
            if content is None:
                return "Error: OpenAI returned empty response"
            return content
        except Exception as e:
            return f"Error with OpenAI: {str(e)}"
