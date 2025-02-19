import streamlit as st
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime
from providers.openai_provider import OpenAIProvider
from providers.google_provider import GoogleProvider
from providers.anthropic_provider import AnthropicProvider
from providers.grok_provider import GrokProvider
from providers.llama_provider import LlamaProvider
from providers.grounded_google_provider import GroundedGoogleProvider
from providers.perplexity_provider import PerplexityProvider
from providers.deepseek_provider import DeepseekProvider
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
        "gemini-2.0-flash-exp": GoogleProvider(),
        "claude-3-5-sonnet-latest": AnthropicProvider(),
        "grok-2-1212": GrokProvider(),
        "llama-v3p1-70b-instruct": LlamaProvider(),
        "gemini-1.5-flash-grounded": GroundedGoogleProvider(),
        "sonar-medium-chat": PerplexityProvider(),
        "deepseek-v3": DeepseekProvider()
    }

async def generate_concurrent_responses(providers, selected_providers, system_prompt, user_prompt, temperature, num_submissions, progress_container, progress_bar, status_containers):
    responses = []
    total_calls = sum(1 for p in selected_providers.values() if p) * num_submissions
    current_call = 0
    providers_per_submission = sum(1 for p in selected_providers.values() if p)

    async def process_provider(provider_name, provider, submission_idx):
        nonlocal current_call
        display_name = st.session_state.custom_names.get(provider_name, provider_name)
        is_google = isinstance(provider, (GoogleProvider, GroundedGoogleProvider))

        if is_google:
            status_msg = f"Querying {display_name}... (Submission {submission_idx + 1}/{num_submissions}, waiting for rate limit)"
        else:
            status_msg = f"Querying {display_name}... (Submission {submission_idx + 1}/{num_submissions})"

        status_containers[provider_name].info(status_msg)

        try:
            response = provider.generate_response(
                system_prompt,
                user_prompt,
                temperature
            )
            responses.append((display_name, response))
            status_containers[provider_name].success(f"{display_name}: Response received (Submission {submission_idx + 1}/{num_submissions})")
        except Exception as e:
            error_msg = f"Error with {display_name}: {str(e)}"
            responses.append((display_name, error_msg))
            status_containers[provider_name].error(error_msg)

        current_call += 1
        progress = current_call / total_calls
        progress_bar.progress(progress)

        # Calculate submission progress including all providers
        submission_progress = (current_call - 1) // providers_per_submission + 1
        progress_container.text(f"Processing submission {submission_progress}/{num_submissions} ({int(progress * 100)}% complete)")

    # Create semaphore for non-Google providers
    general_semaphore = asyncio.Semaphore(3)  # Allow 3 concurrent calls for other providers

    async def rate_limited_provider(provider_name, provider, submission_idx):
        # Google providers already have rate limiting built in
        if isinstance(provider, (GoogleProvider, GroundedGoogleProvider)):
            return await process_provider(provider_name, provider, submission_idx)
        else:
            async with general_semaphore:
                return await process_provider(provider_name, provider, submission_idx)

    tasks = []
    for submission_idx in range(num_submissions):
        for provider_name, provider in providers.items():
            if selected_providers[provider_name]:
                tasks.append(rate_limited_provider(provider_name, provider, submission_idx))

    await asyncio.gather(*tasks)
    return responses

async def async_main():
    st.set_page_config(page_title="LLM Diagnostics", layout="wide")

    # Create tabs for different diagnostic modes
    tab1, tab2 = st.tabs(["🎯 Single Prompt Diagnostics", "🎮 Multi-Prompt Diagnostics"])

    with tab1:
        st.header("Single Prompt Diagnostics")

    with tab2:
        st.header("Multi-Prompt Diagnostics")

        # System prompt for all questions
        system_prompt = st.text_area(
            "System Prompt (applied to all questions)",
            value=st.session_state.get('multi_system_prompt', ''),
            height=150,
            placeholder="Enter system prompt here..."
        )

        # File uploader for CSV and TXT files
        uploaded_file = st.file_uploader("Upload file with questions (CSV or TXT)", type=['csv', 'txt'])

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.txt'):
                    # Read TXT file line by line as questions
                    content = uploaded_file.getvalue().decode('utf-8')
                    questions = [line.strip() for line in content.split('\n') if line.strip()]
                    questions_df = pd.DataFrame({'question': questions})
                    actual_questions = len(questions_df) - 1 if len(questions_df) > 0 else 0
                    st.success(f"Loaded {actual_questions} questions from TXT file")
                else:
                    # Read CSV file
                    questions_df = pd.read_csv(uploaded_file)
                    if 'question' not in questions_df.columns:
                        st.error("CSV file must contain a 'question' column")
                        return
                    st.success(f"Loaded {len(questions_df)} questions from CSV")

                # Add Generate Responses button here after successful file load
                if st.button("Generate Responses", key="multi_prompt_generate"):
                    if not any(selected_providers.values()):
                        st.error("Please select at least one LLM provider")
                        return
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                return

            progress_container = st.empty()
            progress_bar = st.progress(0)
            status_containers = {provider: st.empty() for provider in providers.keys()}
            start_time = time.time()

            try:
                all_responses = []
                total_questions = len(questions_df)
                question_index = 0

                # Process each question multiple times based on num_submissions
                for q_idx, row in questions_df.iterrows():
                    question_index += 1
                    user_prompt = row['question']

                    for submission in range(num_submissions):
                        for provider_name, provider in providers.items():
                            if selected_providers[provider_name]:
                                display_name = st.session_state.custom_names.get(provider_name, provider_name)
                                status_containers[provider_name].info(f"Processing {display_name} - Question {question_index}/{total_questions} (Submission {submission + 1}/{num_submissions})")

                                try:
                                    response = provider.generate_response(
                                        system_prompt,
                                        user_prompt,
                                        temperature
                                    )
                                    all_responses.append((display_name, question_index, response))
                                    status_containers[provider_name].success(f"{display_name}: Response received for Q{question_index}")
                                except Exception as e:
                                    error_msg = f"Error with {display_name}: {str(e)}"
                                    all_responses.append((display_name, question_index, error_msg))
                                    status_containers[provider_name].error(error_msg)

                                # Update progress
                                progress = (question_index * num_submissions + submission) / (total_questions * num_submissions)
                                progress_bar.progress(progress)
                                progress_container.text(f"Processing question {question_index}/{total_questions} (Submission {submission + 1}/{num_submissions})")

                    total_execution_time = time.time() - start_time

                    progress_container.empty()
                    for container in status_containers.values():
                        container.empty()
                    progress_bar.empty()

                    st.info(f"Total execution time: {format_execution_time(total_execution_time)}")

                    if all_responses:
                        # Create filename with timestamp
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"multi_prompt_responses_{timestamp}.csv"

                        # Save responses to CSV
                        df = pd.DataFrame(all_responses, columns=['model', 'q_number', 'response'])
                        df.to_csv(filename, index=False)

                        st.success(f"Responses have been saved to CSV file: {filename}")

                        with open(filename, 'rb') as f:
                            st.download_button(
                                label="Download CSV",
                                data=f,
                                file_name=filename,
                                mime='text/csv'
                            )
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

    # Initialize providers and selected providers
    providers = initialize_providers()
    selected_providers = {} # Moved here

    # Load custom names
    if 'custom_names' not in st.session_state:
        st.session_state.custom_names = load_custom_names()

    # Sidebar controls
    st.sidebar.header("Settings")

    # Model Settings Section
    st.sidebar.subheader("Model Settings")

    # Initialize selected providers and custom names with session state
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

    # Main content for Single Prompt Diagnostics
    with tab1:
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

    if st.button("Generate Responses", key="single_prompt_generate"):
        if not any(selected_providers.values()):
            st.error("Please select at least one LLM provider")
            return

        if not user_prompt:
            st.error("Please enter a user prompt")
            return

        progress_container = st.empty()
        progress_bar = st.progress(0)
        status_containers = {provider: st.empty() for provider in providers.keys()}
        start_time = time.time()

        try:
            responses = await generate_concurrent_responses(
                providers,
                selected_providers,
                system_prompt,
                user_prompt,
                temperature,
                num_submissions,
                progress_container,
                progress_bar,
                status_containers
            )

            total_execution_time = time.time() - start_time

            progress_container.empty()
            for container in status_containers.values():
                container.empty()
            progress_bar.empty()

            st.info(f"Total execution time: {format_execution_time(total_execution_time)}")

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
            progress_container.empty()
            for container in status_containers.values():
                container.empty()
            progress_bar.empty()

if __name__ == "__main__":
    asyncio.run(async_main())