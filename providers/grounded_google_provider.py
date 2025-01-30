import os
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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
        # Configure session with retries
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET"]
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    @property
    def name(self) -> str:
        return "gemini-1.5-flash-grounded"

    def _follow_redirect(self, url: str) -> str:
        """Follow URL redirect with improved error handling and retry logic"""
        try:
            # First try HEAD request with shorter timeout
            response = self.session.head(
                url, 
                allow_redirects=True, 
                timeout=3,
                headers={'User-Agent': 'Mozilla/5.0'}
            )

            if response.status_code == 200:
                return response.url

            # If HEAD fails, try GET
            response = self.session.get(
                url, 
                allow_redirects=True, 
                timeout=5,
                headers={'User-Agent': 'Mozilla/5.0'}
            )

            if response.status_code == 200:
                return response.url

            return url  # Return original if both attempts fail

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
                            if actual_url != chunk.web.uri:  # Only add if redirect was successful
                                urls.append(actual_url)
                            else:
                                print(f"Failed to resolve redirect for: {chunk.web.uri}")

            # Extract avg_logprobs directly from candidate
            avg_logprobs = getattr(candidate, 'avg_logprobs', None)

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