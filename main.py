import streamlit as st

def main():
    # Basic Streamlit configuration
    st.set_page_config(
        page_title="Brandweave LLM Diagnostics",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Basic page elements
    st.title("🤖 Brandweave LLM Diagnostics")
    st.write("Welcome to the LLM Diagnostics Platform!")

    # Simple test elements
    st.sidebar.header("Settings")

    # Test input
    user_input = st.text_input("Enter some text to test", "Hello, world!")
    if user_input:
        st.write("You entered:", user_input)

if __name__ == "__main__":
    main()