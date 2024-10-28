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
            # Combine system and user prompts as Llama uses a single message format
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = self.client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[{"role": "user", "content": combined_prompt}],
                temperature=temperature,
                top_p=0.7,
                max_tokens=1024,
                stream=False  # We don't want streaming for this implementation
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error with Llama: {str(e)}"
