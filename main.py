import streamlit as st
from providers.openai_provider import OpenAIProvider
from providers.google_provider import GoogleProvider
from providers.anthropic_provider import AnthropicProvider
from providers.grok_provider import GrokProvider
from providers.llama_provider import LlamaProvider
from providers.perplexity_provider import PerplexityProvider
from providers.deepseek_provider import DeepseekProvider
from providers.grounded_google_provider import GroundedGoogleProvider
from utils.template_manager import load_templates, save_template, delete_template

def initialize_providers():
    return {
        "gpt-4-turbo": OpenAIProvider(),
        "gemini-1.5-flash": GoogleProvider(),
        "gemini-1.5-flash-grounded": GroundedGoogleProvider(),
        "claude-3-sonnet": AnthropicProvider(),
        "grok-beta": GrokProvider(),
        "llama-v3-70b": LlamaProvider(),
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
    st.write("Welcome to the LLM Diagnostics Platform!")

    # Initialize providers
    providers = initialize_providers()
    templates = load_templates()

    # Sidebar controls
    st.sidebar.header("Settings")

    # Template management
    settings_tab, template_tab = st.sidebar.tabs(["Settings", "Templates"])

    with settings_tab:
        # Model selection
        selected_providers = {}
        for provider_name in providers.keys():
            selected_providers[provider_name] = st.checkbox(
                f"Use {provider_name}",
                value=True
            )

        # Temperature control
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1
        )

    with template_tab:
        # Template selection
        selected_template = st.selectbox(
            "Choose Template",
            ["Custom"] + list(templates.keys())
        )

        # Template management buttons
        if selected_template != "Custom":
            if st.button("Delete Template"):
                delete_template(selected_template)
                st.rerun()

    # Input areas
    col1, col2 = st.columns(2)
    with col1:
        if selected_template != "Custom":
            template_data = templates[selected_template]
            system_prompt = st.text_area(
                "System Prompt",
                value=template_data.get("system_prompt", ""),
                height=150
            )
        else:
            system_prompt = st.text_area(
                "System Prompt",
                height=150
            )
    with col2:
        if selected_template != "Custom":
            template_data = templates[selected_template]
            user_prompt = st.text_area(
                "User Prompt",
                value=template_data.get("user_prompt", ""),
                height=150
            )
        else:
            user_prompt = st.text_area(
                "User Prompt",
                height=150
            )

    # Save template option
    if selected_template == "Custom":
        col1, col2 = st.columns([3, 1])
        with col1:
            new_template_name = st.text_input("Template Name")
        with col2:
            if st.button("Save Template") and new_template_name:
                save_template(
                    new_template_name,
                    system_prompt,
                    user_prompt,
                    selected_providers,
                    temperature
                )
                st.rerun()

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