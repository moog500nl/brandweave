import os
import json
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
                    "dynamic_threshold": 0.5
                }
            }
        }
        self.model = genai.GenerativeModel('gemini-1.5-flash', tools=search_config)
        
    @property
    def name(self) -> str:
        return "gemini-1.5-flash-grounded"

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
            if hasattr(response.candidates[0], 'grounding_metadata'):
                metadata = response.candidates[0].grounding_metadata
                
                # Extract search score if available
                if hasattr(metadata, 'retrieval_metadata'):
                    grounding_data['dynamic_retrieval_score'] = metadata.retrieval_metadata.google_search_dynamic_retrieval_score
                
                # Extract sources if available
                if hasattr(metadata, 'grounding_chunks'):
                    sources = []
                    for chunk in metadata.grounding_chunks:
                        if hasattr(chunk, 'web'):
                            sources.append({
                                'uri': chunk.web.uri,
                                'title': chunk.web.title
                            })
                    grounding_data['sources'] = sources
            
            # Combine response text with metadata
            full_response = {
                'response': response_text,
                'grounding_data': grounding_data
            }
            
            # Return JSON string of the full response
            return json.dumps(full_response, indent=2)
            
        except Exception as e:
            return f"Error with Grounded Google: {str(e)}"
