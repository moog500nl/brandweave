import os
import time
import google.generativeai as genai
from .base import LLMProvider

class RateLimiter:
    def __init__(self, rate=10, per=60):
        self.interval = per / rate  # Time between requests
        self.last_check = 0

    def acquire(self):
        current = time.time()
        wait_time = self.last_check + self.interval - current
        if wait_time > 0:
            time.sleep(wait_time)
        self.last_check = time.time()

class GoogleProvider(LLMProvider):
    def __init__(self):
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.rate_limiter = RateLimiter(rate=10, per=60)  # 10 requests per minute
        
    @property
    def name(self) -> str:
        return "gemini-2.0-flash-exp"

    def generate_response(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            self.rate_limiter.acquire()  # Wait if necessary to respect rate limit
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
