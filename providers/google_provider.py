import os
import google.generativeai as genai
import aiohttp
from .base import LLMProvider

class GoogleProvider(LLMProvider):
    def __init__(self):
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    @property
    def name(self) -> str:
        return "gemini-1.5-flash"

    async def generate_response_async(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            # Google's API doesn't have async support, so we'll use it in a way that doesn't block
            response = await self.model.generate_content_async(
                combined_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature
                )
            )
            return response.text.strip()
        except Exception as e:
            return f"Error with Google: {str(e)}"
