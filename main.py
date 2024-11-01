import streamlit as st
from providers.openai_provider import OpenAIProvider
from providers.google_provider import GoogleProvider
from providers.anthropic_provider import AnthropicProvider
from providers.grok_provider import GrokProvider
from providers.llama_provider import LlamaProvider
from utils.csv_handler import save_responses_to_csv
from utils.template_manager import (
    save_template, get_template, delete_template,
    list_templates, load_custom_names, save_custom_names
)
import time

def format_execution_time(seconds: float) -> str:
    """Format execution time in minutes and seconds"""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    
    if minutes > 0:
        return f"{minutes} minute{'s' if minutes != 1 else ''} and {remaining_seconds} second{'s' if remaining_seconds != 1 else ''}"
    return f"{remaining_seconds} second{'s' if remaining_seconds != 1 else ''}"

def initialize_providers():
    return {
        "gpt-4o-mini": OpenAIProvider(),
        "gemini-1.5-flash": GoogleProvider(),
        "claude-3-sonnet-20240229": AnthropicProvider(),
        "grok-beta": GrokProvider(),
        "llama-v3p1-70b-instruct": LlamaProvider()
    }

def main():
    st.set_page_config(page_title="Brandweave LLM Diagnostics", layout="wide")
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
    
    # Initialize selected providers and custom names with session state
    selected_providers = {}
    show_custom_names = st.sidebar.checkbox("Customize Model Names", 
                                          value=st.session_state.get('show_custom_names', False))
    st.session_state['show_custom_names'] = show_custom_names
    
    # Custom name inputs if enabled
    if show_custom_names:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Custom Model Names")
        for provider_name in providers.keys():
            custom_name = st.sidebar.text_input(
                f"Custom name for {provider_name}",
                value=st.session_state.custom_names.get(provider_name, provider_name),
                key=f"custom_name_{provider_name}"
            )
            st.session_state.custom_names[provider_name] = custom_name
        
        if st.sidebar.button("Save Custom Names"):
            save_custom_names(st.session_state.custom_names)
            st.sidebar.success("Custom names saved!")
    
    # Provider selection checkboxes
    st.sidebar.markdown("---")
    st.sidebar.subheader("Select Models")
    for provider_name in providers.keys():
        display_name = st.session_state.custom_names.get(provider_name, provider_name)
        selected_providers[provider_name] = st.sidebar.checkbox(
            f"Use {display_name}",
            value=st.session_state.get(f'selected_{provider_name}', True)
        )
        st.session_state[f'selected_{provider_name}'] = selected_providers[provider_name]

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get('temperature', 0.7),
        step=0.1
    )
    st.session_state['temperature'] = temperature

    num_submissions = st.sidebar.number_input(
        "Number of submissions",
        min_value=1,
        max_value=1000,
        value=1
    )

    # Template Management in Sidebar
    st.sidebar.header("Template Management")
    template_names = list_templates()
    
    if template_names:
        selected_template = st.sidebar.selectbox(
            "Load Template",
            [""] + template_names,
            index=0
        )
        
        if selected_template:
            if st.sidebar.button("Load Selected Template"):
                template = get_template(selected_template)
                if template:
                    st.session_state['system_prompt'] = template['system_prompt']
                    st.session_state['user_prompt'] = template['user_prompt']
                    # Load model settings if they exist
                    if 'selected_providers' in template:
                        for provider_name, selected in template['selected_providers'].items():
                            st.session_state[f'selected_{provider_name}'] = selected
                    if 'temperature' in template:
                        st.session_state['temperature'] = template['temperature']
                    if 'custom_names' in template:
                        st.session_state.custom_names.update(template['custom_names'])
                    st.rerun()
            
            if st.sidebar.button("Delete Selected Template"):
                if delete_template(selected_template):
                    st.sidebar.success(f"Template '{selected_template}' deleted!")
                    st.rerun()

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        system_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.get('system_prompt', ''),
            height=150,
            placeholder="Enter system prompt here..."
        )

    with col2:
        user_prompt = st.text_area(
            "User Prompt",
            value=st.session_state.get('user_prompt', ''),
            height=150,
            placeholder="Enter user prompt here..."
        )

    # Template Save Section
    col3, col4 = st.columns(2)
    with col3:
        new_template_name = st.text_input("Template Name", placeholder="Enter name to save as template")
        if st.button("Save as Template"):
            if new_template_name and system_prompt and user_prompt:
                if save_template(new_template_name, system_prompt, user_prompt, 
                               selected_providers, temperature, st.session_state.custom_names):
                    st.success(f"Template '{new_template_name}' saved successfully!")
                    st.rerun()
            else:
                st.error("Please provide template name and both prompts")

    if st.button("Generate Responses"):
        if not any(selected_providers.values()):
            st.error("Please select at least one LLM provider")
            return

        if not user_prompt:
            st.error("Please enter a user prompt")
            return

        responses = []
        
        # Create a progress container
        progress_container = st.empty()
        
        # Calculate total number of API calls
        total_calls = sum(1 for p in selected_providers.values() if p) * num_submissions
        progress_bar = st.progress(0)
        current_call = 0
        
        # Create status containers for each provider
        status_containers = {provider: st.empty() for provider in providers.keys()}
        
        # Add execution time tracking
        start_time = time.time()
        
        try:
            for submission_idx in range(num_submissions):
                progress_container.text(f"Processing submission {submission_idx + 1}/{num_submissions}")
                
                for provider_name, provider in providers.items():
                    if selected_providers[provider_name]:
                        # Get custom name for display
                        display_name = st.session_state.custom_names.get(provider_name, provider_name)
                        
                        # Update status for current provider
                        status_containers[provider_name].info(f"Querying {display_name}...")
                        
                        try:
                            response = provider.generate_response(
                                system_prompt,
                                user_prompt,
                                temperature
                            )
                            # Use custom name in responses
                            responses.append((display_name, response))
                            status_containers[provider_name].success(f"{display_name}: Response received")
                        except Exception as e:
                            error_msg = f"Error with {display_name}: {str(e)}"
                            responses.append((display_name, error_msg))
                            status_containers[provider_name].error(error_msg)
                        
                        # Update progress
                        current_call += 1
                        progress_bar.progress(current_call / total_calls)
            
            # Calculate total execution time
            total_execution_time = time.time() - start_time
            
            progress_container.empty()
            for container in status_containers.values():
                container.empty()
            progress_bar.empty()
            
            # Display execution time in formatted string
            st.info(f"Total execution time: {format_execution_time(total_execution_time)}")
            
            # Save to CSV
            if responses:
                filename = save_responses_to_csv(responses, total_execution_time)
                st.success(f"Responses have been saved to CSV file: {filename}")
                
                with open(filename, 'rb') as f:
                    st.download_button(
                        label="Download CSV",
                        data=f,
                        file_name=filename,
                        mime='text/csv'
                    )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            # Clean up progress indicators
            progress_container.empty()
            for container in status_containers.values():
                container.empty()
            progress_bar.empty()

if __name__ == "__main__":
    main()
