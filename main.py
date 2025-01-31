import streamlit as st
from providers.openai_provider import OpenAIProvider
from providers.google_provider import GoogleProvider
from providers.anthropic_provider import AnthropicProvider
from providers.grok_provider import GrokProvider
from providers.llama_provider import LlamaProvider
from providers.perplexity_provider import PerplexityProvider
from providers.deepseek_provider import DeepseekProvider
from providers.grounded_google_provider import GroundedGoogleProvider

# Initialize session state for API query counter
if 'api_queries' not in st.session_state:
    st.session_state.api_queries = 0

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

def get_templates():
    return {
        "General Comparison": {
            "system": "You are a helpful assistant providing detailed comparisons.",
            "user": "Please analyze and compare the following:"
        },
        "Technical Analysis": {
            "system": "You are a technical expert providing in-depth technical analysis.",
            "user": "Please provide a technical analysis of:"
        },
        "Creative Writing": {
            "system": "You are a creative writing assistant helping with content creation.",
            "user": "Please write creatively about:"
        },
        "Brand Analysis": {
            "system": "You are a brand analysis expert providing detailed brand insights.",
            "user": "Please analyze the following brand:"
        }
    }

def main():
    # Page configuration
    st.set_page_config(
        page_title="Brandweave LLM Diagnostics",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Title and API Query Counter
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🤖 Brandweave LLM Diagnostics")
        st.write("Welcome to the LLM Diagnostics Platform!")
    with col2:
        st.metric("API Queries Made", st.session_state.api_queries)

    # Initialize providers
    providers = initialize_providers()
    templates = get_templates()

    # Sidebar controls
    st.sidebar.header("Settings")

    # Template selection
    selected_template = st.sidebar.selectbox(
        "Choose Template",
        ["Custom"] + list(templates.keys())
    )

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
        if selected_template != "Custom":
            system_prompt = st.text_area(
                "System Prompt",
                value=templates[selected_template]["system"],
                height=150
            )
        else:
            system_prompt = st.text_area(
                "System Prompt",
                height=150
            )
    with col2:
        if selected_template != "Custom":
            user_prompt = st.text_area(
                "User Prompt",
                value=templates[selected_template]["user"],
                height=150
            )
        else:
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
                        # Increment API query counter
                        st.session_state.api_queries += 1
                    except Exception as e:
                        st.error(f"Error with {provider_name}: {str(e)}")

if __name__ == "__main__":
    main()