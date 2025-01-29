import os
import json
import requests
import google.generativeai as genai
from .base import LLMProvider

class GroundedGoogleProvider(LLMProvider):
    def __init__(self):
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        # Configure grounded search with dynamic retrieval
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
            return url  # Return original URL if redirect fails

    def _validate_and_process_sources(self, metadata) -> tuple[dict, str]:
        """Validate and process sources from metadata"""
        grounding_data = {}
        error_message = None

        try:
            # Extract search score if available
            if hasattr(metadata, 'retrievalMetadata'):
                grounding_data['dynamic_retrieval_score'] = metadata.retrievalMetadata.googleSearchDynamicRetrievalScore

            # Extract and validate sources
            if hasattr(metadata, 'groundingChunks'):
                if not metadata.groundingChunks:
                    error_message = "No grounding chunks found in response"
                else:
                    sources = []
                    for chunk in metadata.groundingChunks:
                        if not hasattr(chunk, 'web'):
                            continue

                        # Follow redirect to get actual URL
                        actual_url = self._follow_redirect(chunk.web.uri)
                        if actual_url:
                            sources.append({
                                'uri': actual_url,
                                'title': chunk.web.title
                            })

                    if not sources:
                        error_message = "No valid sources found in grounding chunks"
                    grounding_data['sources'] = sources
            else:
                error_message = "No grounding metadata available"

        except Exception as e:
            error_message = f"Error processing sources: {str(e)}"

        return grounding_data, error_message

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

            # Extract the response text
            response_text = response.text.strip()

            # Get grounding metadata if available
            grounding_data = {}
            error_message = None

            if hasattr(response.candidates[0], 'groundingMetadata'):
                grounding_data, error_message = self._validate_and_process_sources(response.candidates[0].groundingMetadata)
            else:
                error_message = "No grounding metadata in response"

            # Combine response text with metadata and any error messages
            full_response = {
                'response': response_text,
                'grounding_data': grounding_data
            }

            if error_message:
                full_response['error'] = error_message

            # Return JSON string of the full response
            return json.dumps(full_response, indent=2)

        except Exception as e:
            return f"Error with Grounded Google: {str(e)}"