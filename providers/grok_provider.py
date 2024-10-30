import os
import aiohttp
from .base import LLMProvider

class GrokProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.environ.get("XAI_API_KEY")
        self.api_url = "https://api.x.ai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
    @property
    def name(self) -> str:
        return "grok-beta"

    async def generate_response_async(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "model": "grok-beta",
                "stream": False,
                "temperature": temperature
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        return f"Error with Grok API: {await response.text()}"
                    
                    data = await response.json()
                    return data['choices'][0]['message']['content']
            
        except Exception as e:
            return f"Error with Grok: {str(e)}"
