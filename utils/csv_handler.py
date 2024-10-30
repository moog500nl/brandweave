import pandas as pd
from datetime import datetime
import os
import csv

def save_responses_to_csv(responses: list[tuple[str, str]]) -> str:
    """
    Save responses to CSV file with datetime-based filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"llm_responses_{timestamp}.csv"
    
    df = pd.DataFrame(responses, columns=['model', 'response'])
    df.to_csv(filename, index=False, quoting=csv.QUOTE_MINIMAL)
    return filename
