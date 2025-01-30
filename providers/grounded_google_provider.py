import os
import json
import requests
import google.generativeai as genai
from .base import LLMProvider

class GroundedGoogleProvider(LLMProvider):
    def __init__(self):
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        search_config = {
            "google_search_retrieval": {
                "dynamic_retrieval_config": {
                    "mode": "MODE_UNSPECIFIED"
                }
            }
        }
        self.model = genai.GenerativeModel('gemini-1.5-flash', tools=search_config)

    @property
    def name(self) -> str:
        return "gemini-1.5-flash-grounded"

    def _follow_redirect(self, url: str) -> str:
        """Follow URL redirect and return the final URL"""
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            return response.url if response.status_code == 200 else url
        except Exception as e:
            print(f"Error following redirect for {url}: {str(e)}")
            return url

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

            # Get candidate and extract text content
            candidate = response.candidates[0]
            content_text = response.text.strip()

            # Get actual URLs from grounding chunks
            urls = []
            if hasattr(candidate, 'grounding_metadata'):
                metadata = candidate.grounding_metadata
                if hasattr(metadata, 'grounding_chunks'):
                    for chunk in metadata.grounding_chunks:
                        if hasattr(chunk, 'web'):
                            actual_url = self._follow_redirect(chunk.web.uri)
                            urls.append(actual_url)

            # Extract avg_logprobs if available
            avg_logprobs = None
            if hasattr(candidate, 'safety_ratings'):
                for rating in candidate.safety_ratings:
                    if hasattr(rating, 'avg_logprobs'):
                        avg_logprobs = rating.avg_logprobs
                        break

            # Return JSON with content, URLs and avg_logprobs
            return json.dumps({
                'text': content_text,
                'urls': urls,
                'avg_logprobs': avg_logprobs
            }, indent=2)

        except Exception as e:
            error_response = {
                'error': str(e),
                'text': None,
                'urls': [],
                'avg_logprobs': None
            }
            return json.dumps(error_response, indent=2)