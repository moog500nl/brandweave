import streamlit as st
import asyncio
import aiohttp
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
        "gemini-2.0-flash": GoogleProvider(),
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

        submission_progress = (current_call - 1) // providers_per_submission + 1
        progress_container.text(f"Processing submission {submission_progress}/{num_submissions} ({int(progress * 100)}% complete)")

    general_semaphore = asyncio.Semaphore(3)

    async def rate_limited_provider(provider_name, provider, submission_idx):
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

async def render_single_prompt():
    providers = initialize_providers()

    if 'custom_names' not in st.session_state:
        st.session_state.custom_names = load_custom_names()

    st.sidebar.header("Settings")
    st.sidebar.subheader("Model Settings")

    selected_providers = {}
    show_custom_names = st.sidebar.checkbox("Customize Model Names", 
                                    value=st.session_state.get('show_custom_names', False))
    st.session_state['show_custom_names'] = show_custom_names


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
        value=st.session_state.get('temperature', 1.0),
        step=0.1
    )
    st.session_state['temperature'] = temperature

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

    col1, col2 = st.columns(2)

    # Add number input in the main content area for single prompt
    num_submissions = st.number_input(
        "Number of submissions",
        min_value=1,
        max_value=1000,
        value=st.session_state.get('single_num_submissions', 1),
        key="single_prompt_num_submissions"
    )
    st.session_state['single_num_submissions'] = num_submissions

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

                # Create two columns for the success message and download button
                col_msg, col_btn = st.columns([2,1])

                with col_msg:
                    st.success(f"✅ Responses saved as: {filename}")
                    st.info("The file will be downloaded to your browser's default downloads folder")

                with col_btn:
                    with open(filename, 'rb') as f:
                        st.download_button(
                            label="📥 Download Results",
                            data=f,
                            file_name=filename,
                            mime='text/csv',
                            use_container_width=True,
                        )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            progress_container.empty()
            for container in status_containers.values():
                container.empty()
            progress_bar.empty()

async def generate_multi_prompt_responses(providers, selected_providers, system_prompt, prompts, temperature, num_submissions, progress_container, progress_bar, status_containers):
    responses = []
    total_calls = len(prompts) * sum(1 for p in selected_providers.values() if p) * num_submissions
    current_call = 0
    providers_per_submission = sum(1 for p in selected_providers.values() if p)

    async def process_provider(provider_name, provider, prompt, q_number, submission_idx):
        nonlocal current_call
        display_name = st.session_state.custom_names.get(provider_name, provider_name)
        is_google = isinstance(provider, (GoogleProvider, GroundedGoogleProvider))

        if is_google:
            status_msg = f"Processing Q{q_number}: {display_name}... (Submission {submission_idx + 1}/{num_submissions}, waiting for rate limit)"
        else:
            status_msg = f"Processing Q{q_number}: {display_name}... (Submission {submission_idx + 1}/{num_submissions})"

        status_containers[provider_name].info(status_msg)

        try:
            response = provider.generate_response(
                system_prompt,
                prompt,
                temperature
            )
            responses.append((display_name, q_number, response))
            status_containers[provider_name].success(f"Q{q_number} - {display_name}: Response received (Submission {submission_idx + 1}/{num_submissions})")
        except Exception as e:
            error_msg = f"Error with {display_name}: {str(e)}"
            responses.append((display_name, q_number, error_msg))
            status_containers[provider_name].error(error_msg)

        current_call += 1
        progress = current_call / total_calls
        progress_bar.progress(progress)

        total_submissions = len(prompts) * num_submissions
        current_submission = ((current_call - 1) // providers_per_submission) + 1
        progress_container.text(f"Processing submission {current_submission}/{total_submissions} ({int(progress * 100)}% complete)")

    general_semaphore = asyncio.Semaphore(3)

    async def rate_limited_provider(provider_name, provider, prompt, q_number, submission_idx):
        if isinstance(provider, (GoogleProvider, GroundedGoogleProvider)):
            return await process_provider(provider_name, provider, prompt, q_number, submission_idx)
        else:
            async with general_semaphore:
                return await process_provider(provider_name, provider, prompt, q_number, submission_idx)

    tasks = []
    for submission_idx in range(num_submissions):
        for q_number, prompt in enumerate(prompts, 1):
            for provider_name, provider in providers.items():
                if selected_providers[provider_name]:
                    tasks.append(rate_limited_provider(provider_name, provider, prompt, q_number, submission_idx))

    await asyncio.gather(*tasks)
    return responses

async def render_multi_prompt():
    providers = initialize_providers()

    if 'custom_names' not in st.session_state:
        st.session_state.custom_names = load_custom_names()

    col1, col2 = st.columns(2)

    with col1:
        system_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.get('multi_system_prompt', ''),
            height=150,
            placeholder="Enter system prompt here...",
            key="multi_system_prompt_input"
        )

    # Add number input between system prompt and file upload
    num_submissions = st.number_input(
        "Number of submissions",
        min_value=1,
        max_value=1000,
        value=st.session_state.get('multi_num_submissions', 1),
        key="multi_prompt_num_submissions"
    )
    st.session_state['multi_num_submissions'] = num_submissions

    uploaded_file = st.file_uploader(
        "Upload a file containing user prompts (CSV or TXT)",
        type=['csv', 'txt'],
        key="multi_prompt_file_uploader"
    )

    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode()
            prompts = [prompt.strip() for prompt in content.splitlines() if prompt.strip()]

            st.subheader("Extracted Prompts")
            for i, prompt in enumerate(prompts, 1):
                st.write(f"{i}. {prompt}")

            st.session_state['multi_prompts'] = prompts
            st.success(f"Successfully loaded {len(prompts)} prompts from the file")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.session_state['multi_prompts'] = []

    if st.button("Generate Responses", key="multi_prompt_generate"):
        selected_providers = {name: st.session_state.get(f'selected_{name}', False) 
                            for name in providers.keys()}

        if not any(selected_providers.values()):
            st.error("Please select at least one LLM provider in the settings")
            return

        if not system_prompt:
            st.error("Please enter a system prompt")
            return

        if not st.session_state.get('multi_prompts'):
            st.error("Please upload a file with prompts")
            return

        progress_container = st.empty()
        progress_bar = st.progress(0)
        status_containers = {provider: st.empty() for provider in providers.keys()}
        start_time = time.time()

        try:
            responses = await generate_multi_prompt_responses(
                providers,
                selected_providers,
                system_prompt,
                st.session_state['multi_prompts'],
                st.session_state.get('temperature', 1.0),
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
                filename = save_responses_to_csv(responses, total_execution_time, is_multi_prompt=True)

                # Create two columns for the success message and download button
                col_msg, col_btn = st.columns([2,1])

                with col_msg:
                    st.success(f"✅ Responses saved as: {filename}")
                    st.info("The file will be downloaded to your browser's default downloads folder")

                with col_btn:
                    with open(filename, 'rb') as f:
                        st.download_button(
                            label="📥 Download Results",
                            data=f,
                            file_name=filename,
                            mime='text/csv',
                            use_container_width=True,
                        )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            progress_container.empty()
            for container in status_containers.values():
                container.empty()
            progress_bar.empty()

async def async_main():
    st.set_page_config(page_title="Brandweave LLM Diagnostics", layout="wide")
    st.title("🤖 Brandweave LLM Diagnostics")

    tab1, tab2 = st.tabs(["Single Prompt", "Multi-prompt"])

    with tab1:
        await render_single_prompt()

    with tab2:
        await render_multi_prompt()

if __name__ == "__main__":
    asyncio.run(async_main())