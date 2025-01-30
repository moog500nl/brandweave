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
                    "mode": "MODE_DYNAMIC",
                    "dynamic_threshold": 0.0
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

            # Build raw response object
            raw_response = {
                'response': response.text.strip(),
                'grounding_data': {}
            }

            # Extract metadata if available
            if hasattr(response.candidates[0], 'groundingMetadata'):
                metadata = response.candidates[0].groundingMetadata

                # Get retrieval score
                if hasattr(metadata, 'retrievalMetadata'):
                    raw_response['grounding_data']['dynamic_retrieval_score'] = (
                        metadata.retrievalMetadata.googleSearchDynamicRetrievalScore
                    )

                # Extract and follow redirect URLs
                if hasattr(metadata, 'groundingChunks'):
                    sources = []
                    for chunk in metadata.groundingChunks:
                        if hasattr(chunk, 'web'):
                            actual_url = self._follow_redirect(chunk.web.uri)
                            sources.append({
                                'uri': actual_url,
                                'title': chunk.web.title
                            })
                    raw_response['grounding_data']['sources'] = sources

            # Return JSON string
            return json.dumps(raw_response, indent=2)

        except Exception as e:
            error_response = {
                'error': f"Error with Grounded Google: {str(e)}",
                'response': None,
                'grounding_data': {}
            }
            return json.dumps(error_response, indent=2)