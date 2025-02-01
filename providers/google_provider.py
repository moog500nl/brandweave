import os
import google.generativeai as genai
from .base import LLMProvider

class GoogleProvider(LLMProvider):
    def __init__(self):
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
    @property
    def name(self) -> str:
        return "gemini-2.0-flash-exp"

    def generate_response(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.model.generate_content(
                combined_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=1000           
                )
            )
            return response.text.strip()
        except Exception as e:
            return f"Error with Google: {str(e)}"
