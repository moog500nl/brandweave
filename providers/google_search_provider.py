import os
import google.generativeai as genai
from .base import LLMProvider

class GoogleSearchProvider(LLMProvider):
    def __init__(self):
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    @property
    def name(self) -> str:
        return "gemini-1.5-flash-search"

    def generate_response(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        try:
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"

            # Configure the generation parameters including search
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=1000,
                candidate_count=1
            )

            # Generate content with search enabled
            response = self.model.generate_content(
                combined_prompt,
                generation_config=generation_config,
                stream=False
            )

            # Extract the main response
            content = response.text.strip()

            # Get the search quality score if available
            search_score = "N/A"
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'search_quality_score'):
                        search_score = str(candidate.search_quality_score)
                        break

            # Return both content and search score
            return f"{content}\nSearch Quality Score: {search_score}"

        except Exception as e:
            return f"Error with Google Search: {str(e)}"