import pandas as pd
import plotly.express as px
from collections import Counter
from typing import List, Tuple

def analyze_brand_frequencies(responses: List[Tuple[str, str]]) -> dict:
    """
    Analyze brand frequencies from responses
    Returns a dictionary containing visualization data
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
    
    # Convert to DataFrame for visualization
    freq_df = pd.DataFrame.from_dict(brand_counts, orient='index', columns=['count'])
    freq_df = freq_df.reset_index().rename(columns={'index': 'brand'})
    freq_df = freq_df.sort_values('count', ascending=True)
    
    # Create bar chart
    fig = px.bar(freq_df, 
                 x='count', 
                 y='brand',
                 orientation='h',
                 title='Brand Frequency Analysis',
                 labels={'count': 'Frequency', 'brand': 'Brand Name'})
    
    # Update layout for better readability
    fig.update_layout(
        height=max(400, len(freq_df) * 25),  # Dynamic height based on number of brands
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    return fig
