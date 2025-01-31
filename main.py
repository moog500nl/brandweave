import streamlit as st
import os
import time
from utils.csv_handler import save_responses_to_csv
from utils.template_manager import (
    save_template, get_template, delete_template,
    list_templates, load_custom_names, save_custom_names
)
from providers.openai_provider import OpenAIProvider
from providers.google_provider import GoogleProvider
from providers.anthropic_provider import AnthropicProvider
from providers.grok_provider import GrokProvider
from providers.llama_provider import LlamaProvider
from providers.perplexity_provider import PerplexityProvider
from providers.deepseek_provider import DeepseekProvider
from providers.grounded_google_provider import GroundedGoogleProvider


def format_execution_time(seconds: float) -> str:
    """Format execution time in minutes and seconds"""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)

    if minutes > 0:
        return f"{minutes} minute{'s' if minutes != 1 else ''} and {remaining_seconds} second{'s' if remaining_seconds != 1 else ''}"
    return f"{remaining_seconds} second{'s' if remaining_seconds != 1 else ''}"

def initialize_providers():
    # Start with empty providers for testing
    return {}

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
    st.write("Testing basic Streamlit functionality...")

if __name__ == "__main__":
    main()