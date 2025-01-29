import os
import json
import google.generativeai as genai
from .base import LLMProvider

class RawGroundedGoogleProvider(LLMProvider):
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
        return "gemini-1.5-flash-raw-grounded"

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

            # Return raw response as JSON string
            raw_metadata = {}
            grounding_metadata = response.candidates[0].grounding_metadata

            if hasattr(grounding_metadata, 'grounding_chunks'):
                raw_metadata['grounding_chunks'] = [
                    {'web': {'uri': chunk.web.uri, 'title': chunk.web.title}} 
                    for chunk in grounding_metadata.grounding_chunks if hasattr(chunk, 'web')
                ]

            if hasattr(grounding_metadata, 'web_search_queries'):
                raw_metadata['web_search_queries'] = grounding_metadata.web_search_queries

            if hasattr(grounding_metadata, 'grounding_supports'):
                raw_metadata['grounding_supports'] = [
                    {
                        'segment': {
                            'start_index': support.segment.start_index,
                            'end_index': support.segment.end_index,
                            'text': support.segment.text
                        },
                        'grounding_chunk_indices': support.grounding_chunk_indices,
                        'confidence_scores': support.confidence_scores
                    }
                    for support in grounding_metadata.grounding_supports
                ]

            return json.dumps({
                'response': response.text.strip(),
                'metadata': raw_metadata
            }, indent=2)

        except Exception as e:
            return f"Error with Raw Grounded Google: {str(e)}"