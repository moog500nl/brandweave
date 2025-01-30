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

            # Initialize the basic response structure
            output = {
                "response": response.text.strip()
            }

            # Process metadata if available
            if hasattr(response.candidates[0], 'grounding_metadata'):
                metadata = response.candidates[0].grounding_metadata

                # Initialize grounding_data structure
                grounding_data = {}

                # Extract retrieval score if available
                if hasattr(metadata, 'retrieval_metadata'):
                    grounding_data['dynamic_retrieval_score'] = (
                        metadata.retrieval_metadata.google_search_dynamic_retrieval_score
                    )

                # Extract sources and follow redirects
                if hasattr(metadata, 'grounding_chunks'):
                    sources = []
                    for chunk in metadata.grounding_chunks:
                        if hasattr(chunk, 'web'):
                            # Follow redirect to get actual URL
                            actual_url = self._follow_redirect(chunk.web.uri)
                            sources.append({
                                'uri': actual_url,
                                'title': chunk.web.title
                            })
                    if sources:
                        grounding_data['sources'] = sources

                # Add grounding data if we collected any
                if grounding_data:
                    output['grounding_data'] = grounding_data

            # Convert to CSV-friendly format
            csv_output = {
                "Response": output['response'],
                "RetrievalScore": (
                    output.get('grounding_data', {}).get('dynamic_retrieval_score', "")
                ),
                "URLs": "; ".join(
                    source['uri'] 
                    for source in output.get('grounding_data', {}).get('sources', [])
                )
            }

            return json.dumps(csv_output)

        except Exception as e:
            return f"Error with Grounded Google: {str(e)}"