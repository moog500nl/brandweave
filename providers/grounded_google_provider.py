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

    def _extract_urls_from_sources(self, sources) -> list:
        """Extract and follow redirects for URLs from sources"""
        urls = []
        for source in sources:
            if 'uri' in source:
                actual_url = self._follow_redirect(source['uri'])
                urls.append(actual_url)
        return urls

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

            # Get the candidate response
            candidate = response.candidates[0]
            response_text = response.text.strip()

            # Initialize the output structure
            output = {
                "response": response_text,
                "dynamic_retrieval_score": None,
                "urls": []
            }

            # Extract metadata if available
            if hasattr(candidate, 'grounding_metadata'):
                metadata = candidate.grounding_metadata

                # Get dynamic retrieval score
                if hasattr(metadata, 'retrieval_metadata'):
                    output['dynamic_retrieval_score'] = metadata.retrieval_metadata.google_search_dynamic_retrieval_score

                # Extract and follow URLs from sources
                if hasattr(metadata, 'grounding_chunks'):
                    for chunk in metadata.grounding_chunks:
                        if hasattr(chunk, 'web'):
                            actual_url = self._follow_redirect(chunk.web.uri)
                            output['urls'].append(actual_url)

            # Convert the output to a CSV-friendly format
            # URLs will be joined with semicolons to keep them in one column
            csv_output = {
                "Response": output['response'],
                "RetrievalScore": output['dynamic_retrieval_score'] if output['dynamic_retrieval_score'] is not None else "",
                "URLs": "; ".join(output['urls']) if output['urls'] else ""
            }

            return json.dumps(csv_output)

        except Exception as e:
            return f"Error with Grounded Google: {str(e)}"