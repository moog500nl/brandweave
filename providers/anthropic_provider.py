import os
import anthropic
from .base import LLMProvider

class AnthropicProvider(LLMProvider):
    def __init__(self, model_name: str = "claude-3-sonnet-20240229"):
        super().__init__(model_name)
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
    def generate_response(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            message = self.client.messages.create(
                model=self._model_name,
                max_tokens=1000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature
            )
            if not message.content:
                return "Error: Anthropic returned empty response"
            return message.content[0].text

        except Exception as e:
            return f"Error with Anthropic: {str(e)}"
