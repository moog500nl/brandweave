import os
from openai import AsyncOpenAI
from .base import LLMProvider

class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    @property
    def name(self) -> str:
        return "gpt-4o-mini"

    async def generate_response_async(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            content = response.choices[0].message.content
            if content is None:
                return "Error: OpenAI returned empty response"
            return content
        except Exception as e:
            return f"Error with OpenAI: {str(e)}"
