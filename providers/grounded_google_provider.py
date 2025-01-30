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

            # Get the first candidate's response
            candidate = response.candidates[0]

            # Build raw debug response with all metadata
            raw_response = {
                'response': response.text.strip(),
                'raw_metadata': {},
                'debug_info': {}
            }

            # Add grounding metadata if available
            if hasattr(candidate, 'groundingMetadata'):
                metadata = candidate.groundingMetadata
                raw_response['debug_info']['has_grounding_metadata'] = True

                # Log all available attributes
                for attr in dir(metadata):
                    if not attr.startswith('_'):
                        try:
                            value = getattr(metadata, attr)
                            raw_response['raw_metadata'][attr] = str(value)
                        except Exception as attr_error:
                            raw_response['debug_info'][f'error_getting_{attr}'] = str(attr_error)
            else:
                raw_response['debug_info']['has_grounding_metadata'] = False

            # Build CSV-friendly processed response
            processed_response = {
                "Response": response.text.strip(),
                "RetrievalScore": "",
                "URLs": ""
            }

            # Process metadata for CSV format if available
            if hasattr(candidate, 'groundingMetadata'):
                metadata = candidate.groundingMetadata

                # Get retrieval score
                if hasattr(metadata, 'retrievalMetadata'):
                    processed_response['RetrievalScore'] = str(
                        metadata.retrievalMetadata.googleSearchDynamicRetrievalScore
                    )

                # Get and follow redirect URLs
                if hasattr(metadata, 'groundingChunks'):
                    urls = []
                    for chunk in metadata.groundingChunks:
                        if hasattr(chunk, 'web'):
                            actual_url = self._follow_redirect(chunk.web.uri)
                            urls.append(actual_url)
                    processed_response['URLs'] = "; ".join(urls)

            # Return both responses as a dictionary
            return json.dumps({
                'raw_response': raw_response,
                'processed_response': processed_response
            }, indent=2)

        except Exception as e:
            error_response = {
                'raw_response': {'error': f"Error with Grounded Google: {str(e)}"},
                'processed_response': {'error': f"Error with Grounded Google: {str(e)}"}
            }
            return json.dumps(error_response, indent=2)