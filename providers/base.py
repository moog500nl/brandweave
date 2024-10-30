from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    async def generate_response_async(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        pass

    def generate_response(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        """Synchronous wrapper for backward compatibility"""
        import asyncio
        return asyncio.run(self.generate_response_async(system_prompt, user_prompt, temperature))

    @property
    @abstractmethod
    def name(self) -> str:
        pass
