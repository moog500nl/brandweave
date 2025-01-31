import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.debug("Starting Streamlit application")

        # Basic Streamlit configuration from original code
        st.set_page_config(
            page_title="Brandweave LLM Diagnostics",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        logger.debug("Page config set")

        # Basic page elements from original code
        st.title("🤖 Brandweave LLM Diagnostics")
        st.write("Welcome to the LLM Diagnostics Platform!")

        # Simple test elements from original code
        st.sidebar.header("Settings")

        # Test input from original code
        user_input = st.text_input("Enter some text to test", "Hello, world!")
        if user_input:
            st.write("You entered:", user_input)

        logger.debug("UI elements added")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    logger.debug("Script started")
    main()
    logger.debug("Script completed")