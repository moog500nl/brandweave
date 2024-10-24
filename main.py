import streamlit as st
from providers.openai_provider import OpenAIProvider
from providers.google_provider import GoogleProvider
from providers.anthropic_provider import AnthropicProvider
from utils.csv_handler import save_responses_to_csv

def initialize_providers():
    return {
        "gpt-4o-mini": OpenAIProvider(),
        "gemini-1.5-flash": GoogleProvider(),
        "claude-3-sonnet-20240229": AnthropicProvider()
    }

def main():
    st.set_page_config(page_title="LLM Comparison Tool", layout="wide")
    st.title("🤖 LLM Comparison Tool")

    # Initialize providers
    providers = initialize_providers()

    # Sidebar controls
    st.sidebar.header("Settings")
    
    selected_providers = {}
    for provider_name in providers.keys():
        selected_providers[provider_name] = st.sidebar.checkbox(
            f"Use {provider_name}",
            value=True
        )

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )

    num_submissions = st.sidebar.number_input(
        "Number of submissions",
        min_value=1,
        max_value=10,
        value=1
    )

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        system_prompt = st.text_area(
            "System Prompt",
            height=150,
            placeholder="Enter system prompt here..."
        )

    with col2:
        user_prompt = st.text_area(
            "User Prompt",
            height=150,
            placeholder="Enter user prompt here..."
        )

    if st.button("Generate Responses"):
        if not any(selected_providers.values()):
            st.error("Please select at least one LLM provider")
            return

        if not user_prompt:
            st.error("Please enter a user prompt")
            return

        responses = []
        
        with st.spinner("Generating responses..."):
            for _ in range(num_submissions):
                for provider_name, provider in providers.items():
                    if selected_providers[provider_name]:
                        response = provider.generate_response(
                            system_prompt,
                            user_prompt,
                            temperature
                        )
                        responses.append((provider.name, response))

        # Display responses
        st.subheader("Responses")
        for provider_name, response in responses:
            with st.expander(f"Response from {provider_name}"):
                st.write(response)

        # Save to CSV
        if responses:
            filename = save_responses_to_csv(responses)
            st.success(f"Responses saved to {filename}")
            
            with open(filename, 'rb') as f:
                st.download_button(
                    label="Download CSV",
                    data=f,
                    file_name=filename,
                    mime='text/csv'
                )

if __name__ == "__main__":
    main()
