async def render_multi_prompt():
    providers = initialize_providers()

    if 'custom_names' not in st.session_state:
        st.session_state.custom_names = load_custom_names()

    # Add number_input for submissions
    num_submissions = st.sidebar.number_input(
        "Number of submissions",
        min_value=1,
        max_value=1000,
        value=st.session_state.get('num_submissions', 1),
        key="multi_prompt_num_submissions"  # Different key from single prompt
    )
    
    # Rest of the function...
    
    if st.button("Generate Responses", key="multi_prompt_generate"):
        # Use the num_submissions from the input
        responses = await generate_multi_prompt_responses(
            providers,
            selected_providers,
            system_prompt,
            st.session_state['multi_prompts'],
            st.session_state.get('temperature', 1.0),
            num_submissions,  # Now using the actual input value
            progress_container,
            progress_bar,
            status_containers
        )
