from abc import ABC, abstractmethod

class LLMProvider(ABC):
    def __init__(self, model_name: str = None):
        self._model_name = model_name
        
    @abstractmethod
    def generate_response(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        pass

    @property
    def name(self) -> str:
        return self._model_name
