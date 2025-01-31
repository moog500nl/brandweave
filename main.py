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
from utils.csv_handler import save_responses_to_csv

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

        # Number of repetitions
        num_repetitions = st.number_input(
            "Number of times to repeat prompt",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            help="Each selected LLM will receive the prompt this many times"
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

        # List to store all responses for CSV
        all_responses = []

        # Generate responses for each provider
        for provider_name, provider in providers.items():
            if selected_providers[provider_name]:
                st.write(f"### {provider_name}")

                # Generate multiple responses per provider
                for i in range(num_repetitions):
                    with st.spinner(f"Querying {provider_name} (Attempt {i+1}/{num_repetitions})..."):
                        try:
                            response = provider.generate_response(
                                system_prompt,
                                user_prompt,
                                temperature
                            )
                            # Add response to the display with attempt number if multiple attempts
                            if num_repetitions > 1:
                                st.write(f"Attempt {i+1}:")
                            st.write(response)

                            # Add to responses list for CSV
                            all_responses.append((f"{provider_name}_attempt_{i+1}", response))

                        except Exception as e:
                            error_msg = f"Error with {provider_name}: {str(e)}"
                            st.error(error_msg)
                            all_responses.append((f"{provider_name}_attempt_{i+1}", error_msg))

        # Save all responses to CSV
        if all_responses:
            csv_filename = save_responses_to_csv(all_responses)
            st.success(f"Responses saved to {csv_filename}")

if __name__ == "__main__":
    main()