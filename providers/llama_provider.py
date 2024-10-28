import os
from openai import OpenAI
from .base import LLMProvider

class LlamaProvider(LLMProvider):
    def __init__(self):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ.get("NVIDIA_API_KEY")
        )
        
    @property
    def name(self) -> str:
        return "llama-3.1-70b-instruct"

    def generate_response(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            messages = [
                {"role": "system", "content": system_prompt} if system_prompt else None,
                {"role": "user", "content": user_prompt}
            ]
            # Filter out None messages if system prompt is empty
            messages = [msg for msg in messages if msg is not None]
            
            response = self.client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=messages,
                temperature=temperature,
                top_p=0.7,
                max_tokens=1024,
                stream=True
            )
            
            # Handle streaming response
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    
            return full_response.strip()
            
        except Exception as e:
            return f"Error with Llama: {str(e)}"
