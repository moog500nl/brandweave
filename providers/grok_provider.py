import os
import requests
from .base import LLMProvider

class GrokProvider(LLMProvider):
    def __init__(self, model_name: str = "grok-beta"):
        super().__init__(model_name)
        self.api_key = os.environ.get("XAI_API_KEY")
        self.api_url = "https://api.x.ai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
    def generate_response(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "model": self._model_name,
                "stream": False,
                "temperature": temperature,
                "max_tokens": 1000
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code != 200:
                return f"Error with Grok API: {response.text}"
                
            data = response.json()
            return data['choices'][0]['message']['content']
            
        except Exception as e:
            return f"Error with Grok: {str(e)}"
