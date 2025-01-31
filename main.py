import streamlit as st
from providers.openai_provider import OpenAIProvider
from providers.google_provider import GoogleProvider
from providers.anthropic_provider import AnthropicProvider
from providers.grok_provider import GrokProvider
from providers.llama_provider import LlamaProvider
from providers.perplexity_provider import PerplexityProvider
from providers.deepseek_provider import DeepseekProvider
from providers.grounded_google_provider import GroundedGoogleProvider
from utils.csv_handler import save_responses_to_csv
from utils.template_manager import (
    save_template, get_template, delete_template,
    list_templates, load_custom_names, save_custom_names
)
import time

def format_execution_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    if minutes > 0:
        return f"{minutes} minute{'s' if minutes != 1 else ''} and {remaining_seconds} second{'s' if remaining_seconds != 1 else ''}"
    return f"{remaining_seconds} second{'s' if remaining_seconds != 1 else ''}"

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
    # Replit-specific configuration
    st.set_page_config(
        page_title="Brandweave LLM Diagnostics",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Basic page elements for testing
    st.title("🤖 Brandweave LLM Diagnostics")
    

    # Initialize providers
    providers = initialize_providers()

    # Load custom names
    if 'custom_names' not in st.session_state:
        st.session_state.custom_names = load_custom_names()

    # Sidebar controls
    st.sidebar.header("Settings")
    

    # Model Settings Section
    st.sidebar.subheader("Model Settings")
    selected_providers = {}

    # Provider selection checkboxes
    for provider_name in providers.keys():
        display_name = st.session_state.custom_names.get(provider_name, provider_name)
        selected_providers[provider_name] = st.sidebar.checkbox(
            f"Use {display_name}",
            value=st.session_state.get(f'selected_{provider_name}', True)
        )
        st.session_state[f'selected_{provider_name}'] = selected_providers[provider_name]

    # Temperature slider
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get('temperature', 0.7),
        step=0.1
    )
    st.session_state['temperature'] = temperature

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        system_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.get('system_prompt', ''),
            height=150
        )

    with col2:
        user_prompt = st.text_area(
            "User Prompt",
            value=st.session_state.get('user_prompt', ''),
            height=150
        )

    if st.button("Generate Responses"):
        if not any(selected_providers.values()):
            st.error("Please select at least one LLM provider")
            return

        if not user_prompt:
            st.error("Please enter a user prompt")
            return

        responses = []
        progress_bar = st.progress(0)

        try:
            for provider_name, provider in providers.items():
                if selected_providers[provider_name]:
                    display_name = st.session_state.custom_names.get(provider_name, provider_name)
                    with st.spinner(f"Querying {display_name}..."):
                        try:
                            response = provider.generate_response(
                                system_prompt,
                                user_prompt,
                                temperature
                            )
                            responses.append((display_name, response))
                        except Exception as e:
                            error_msg = f"Error with {display_name}: {str(e)}"
                            responses.append((display_name, error_msg))
                            st.error(error_msg)

            if responses:
                filename = save_responses_to_csv(responses, time.time())
                st.success(f"Responses saved to: {filename}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()