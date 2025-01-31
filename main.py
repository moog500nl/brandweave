import streamlit as st

def main():
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

if __name__ == "__main__":
    main()