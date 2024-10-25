import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict, Any
import os
import json
import streamlit.components.v1 as components

def analyze_brand_frequencies(responses: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """
    Analyze brand frequencies from responses and return data for React visualization
    """
    # Convert responses to DataFrame
    df = pd.DataFrame(responses, columns=['model', 'response'])
    
    # Process responses and count brand frequencies
    all_brands = []
    for response in df['response']:
        # Split by comma and strip whitespace
        brands = [brand.strip() for brand in response.split(',')]
        all_brands.extend(brands)
    
    # Count frequencies
    brand_counts = Counter(all_brands)
    
    # Convert to list of dictionaries for React
    data = [{"brand": brand, "count": count} 
            for brand, count in sorted(brand_counts.items(), 
                                    key=lambda x: x[1],
                                    reverse=True)]
    
    return data

def render_brand_chart(data: List[Dict[str, Any]]):
    """
    Render the React-based brand chart
    """
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Path to the frontend build directory
    build_dir = os.path.join(current_dir, "..", "frontend", "build", "static", "js")
    js_path = os.path.join(build_dir, "main.js")
    
    # Create a custom component
    components.html(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap" rel="stylesheet">
            <style>
                #root {{ font-family: 'Roboto', sans-serif; }}
            </style>
        </head>
        <body>
            <div id="root"></div>
            <script>
                window.brandChartData = {json.dumps(data)};
            </script>
            <script src="frontend/build/static/js/main.js"></script>
        </body>
        </html>
        """,
        height=max(400, len(data) * 25),
    )
