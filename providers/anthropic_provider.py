import os
import anthropic
from .base import LLMProvider

class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
    @property
    def name(self) -> str:
        return "claude-3-5-sonnet-latest"

    def generate_response(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-latest",
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
