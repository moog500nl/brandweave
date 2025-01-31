import streamlit as st

# Enable streamlit debug mode
import logging
logging.getLogger('streamlit').setLevel(logging.DEBUG)

def main():
    try:
        st.set_page_config(
            page_title="Brandweave LLM Diagnostics",
            layout="wide",
            menu_items={
                'Get Help': None,
                'Report a bug': None,
                'About': None
            }
        )

        st.title("🤖 Brandweave LLM Diagnostics")
        st.write("Welcome to the LLM Diagnostics Platform")
        st.write("Debug: Page loaded successfully")
    except Exception as e:
        st.error(f"Error loading page: {str(e)}")

if __name__ == "__main__":
    main()