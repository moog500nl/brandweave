import pandas as pd
from datetime import datetime
import os
import csv

def save_responses_to_csv(responses: list[tuple[str, str]], execution_time: float = None, is_multi_prompt: bool = False) -> str:
    """
    Save responses to CSV file with datetime-based filename
    For single prompt: responses = [(model, response), ...]
    For multi prompt: responses = [(model, q_number, response), ...]
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"llm_responses_{timestamp}.csv"

    if is_multi_prompt:
        # Create DataFrame with responses including question number
        df = pd.DataFrame(responses, columns=['model', 'q_number', 'response'])
    else:
        # Create DataFrame with responses only (backward compatibility)
        df = pd.DataFrame(responses, columns=['model', 'response'])

    # Save to CSV
    df.to_csv(filename, index=False, quoting=csv.QUOTE_MINIMAL)
    return filename