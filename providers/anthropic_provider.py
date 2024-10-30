import os
from anthropic import AsyncAnthropic
from .base import LLMProvider

class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
    @property
    def name(self) -> str:
        return "claude-3-sonnet-20240229"

    async def generate_response_async(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            message = await self.client.messages.create(
                model="claude-3-sonnet-20240229",
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
