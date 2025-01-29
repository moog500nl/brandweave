import os
import google.generativeai as genai
from .base import LLMProvider

class GoogleGroundedProvider(LLMProvider):
    def __init__(self):
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(
            'gemini-1.5-flash-002',
            tools="google_search_retrieval"
        )
        
    @property
    def name(self) -> str:
        return "gemini-grounded-search"

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
            
            # Get the first candidate response
            rc = response.candidates[0]
            
            # Get the dynamic retrieval score
            score = rc.grounding_metadata.retrieval_metadata.google_search_dynamic_retrieval_score
            
            # Get the grounding chunks (sources)
            sources = []
            if hasattr(rc, 'grounding_metadata') and hasattr(rc.grounding_metadata, 'grounding_chunks'):
                for chunk in rc.grounding_metadata.grounding_chunks:
                    if hasattr(chunk, 'web'):
                        sources.append({
                            'title': chunk.web.title,
                            'url': chunk.web.uri
                        })

            # Format the response with sources
            formatted_response = response.text.strip()
            if sources:
                formatted_response += "\n\nSources:"
                for idx, source in enumerate(sources, 1):
                    formatted_response += f"\n[{idx}] {source['title']} - {source['url']}"
                
            formatted_response += f"\n\nDynamic Retrieval Score: {score}"
            
            return formatted_response

        except Exception as e:
            return f"Error with Google Grounded Search: {str(e)}"
