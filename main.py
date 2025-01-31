import streamlit as st
from providers.openai_provider import OpenAIProvider
from providers.google_provider import GoogleProvider
from providers.anthropic_provider import AnthropicProvider
from providers.grok_provider import GrokProvider
from providers.llama_provider import LlamaProvider
from providers.perplexity_provider import PerplexityProvider
from providers.deepseek_provider import DeepseekProvider
from providers.grounded_google_provider import GroundedGoogleProvider

def initialize_providers():
    return {
        "gpt-4o-mini": OpenAIProvider(),
        "gemini-1.5-flash": GoogleProvider(),
        "gemini-1.5-flash-grounded": GroundedGoogleProvider(),
        "claude-3-5-sonnet-latest": AnthropicProvider(),
        "grok-beta": GrokProvider(),
        "llama-v3p1-70b-instruct": LlamaProvider(),
        "sonar-medium-chat": PerplexityProvider(),
        "deepseek-v3": DeepseekProvider()
    }

def main():
    # Page configuration
    st.set_page_config(
        page_title="Brandweave LLM Diagnostics",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Title
    st.title("🤖 Brandweave LLM Diagnostics")

    # Initialize providers
    providers = initialize_providers()

    # Sidebar controls
    st.sidebar.header("Settings")

    # Model selection
    selected_providers = {}
    for provider_name in providers.keys():
        selected_providers[provider_name] = st.sidebar.checkbox(
            f"Use {provider_name}",
            value=True
        )

    # Temperature control
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )

    # Input areas
    col1, col2 = st.columns(2)
    with col1:
        system_prompt = st.text_area(
            "System Prompt",
            height=150
        )
    with col2:
        user_prompt = st.text_area(
            "User Prompt",
            height=150
        )

    # Generate button
    if st.button("Generate Responses"):
        if not any(selected_providers.values()):
            st.error("Please select at least one LLM provider")
            return

        if not user_prompt:
            st.error("Please enter a user prompt")
            return

        # Generate responses
        for provider_name, provider in providers.items():
            if selected_providers[provider_name]:
                with st.spinner(f"Querying {provider_name}..."):
                    try:
                        response = provider.generate_response(
                            system_prompt,
                            user_prompt,
                            temperature
                        )
                        st.write(f"### {provider_name}")
                        st.write(response)
                    except Exception as e:
                        st.error(f"Error with {provider_name}: {str(e)}")

if __name__ == "__main__":
    main()