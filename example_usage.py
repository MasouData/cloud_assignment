#!/usr/bin/env python3
"""
Example usage of the optimized HSP Lazy Greedy algorithm
"""

import numpy as np
import pandas as pd
import time
from hsp_optimized import hsp_lazy_greedy_auto

def create_example_data(n=1000000):
    """Create example dataset for demonstration"""
    print(f"Creating example dataset with n={n:,} elements...")
    
    # Create a dataframe with various features
    np.random.seed(42)  # For reproducible results
    df = pd.DataFrame({
        'id': range(n),
        'score': np.random.uniform(0, 100, n),
        'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'value': np.random.exponential(10, n)
    })
    
    return df

def create_example_views(df):
    """Create example views (queries) on the dataset"""
    print("Creating example views...")
    
    views = {}
    
    # High-score elements
    views['high_score'] = lambda df: df[df['score'] > 80]
    
    # Category-based views
    for cat in ['A', 'B', 'C', 'D', 'E']:
        views[f'category_{cat}'] = lambda df, c=cat: df[df['category'] == c]
    
    # Region-based views
    for region in ['North', 'South', 'East', 'West']:
        views[f'region_{region}'] = lambda df, r=region: df[df['region'] == r]
    
    # Value-based views
    views['high_value'] = lambda df: df[df['value'] > 20]
    views['low_value'] = lambda df: df[df['value'] < 5]
    
    # Combined views
    views['high_score_north'] = lambda df: df[(df['score'] > 70) & (df['region'] == 'North')]
    views['category_A_high_value'] = lambda df: df[(df['category'] == 'A') & (df['value'] > 15)]
    
    print(f"Created {len(views)} views")
    return views

def run_example():
    """Run the complete example"""
    print("HSP Lazy Greedy Algorithm - Example Usage")
    print("=" * 50)
    
    # Parameters
    n = 1000000  # Dataset size
    budget = 100  # Number of elements to select
    beta = 0.1   # Penalty parameter
    
    # Create data
    df = create_example_data(n)
    views = create_example_views(df)
    
    # Print dataset info
    print(f"\nDataset Info:")
    print(f"  Elements (n): {len(df):,}")
    print(f"  Views (m): {len(views)}")
    print(f"  Budget: {budget}")
    print(f"  Beta: {beta}")
    
    # Show view coverage
    print(f"\nView Coverage:")
    total_elements = len(df)
    for view_name, view_func in list(views.items())[:5]:  # Show first 5 views
        coverage = len(view_func(df))
        percentage = (coverage / total_elements) * 100
        print(f"  {view_name}: {coverage:,} elements ({percentage:.1f}%)")
    if len(views) > 5:
        print(f"  ... and {len(views) - 5} more views")
    
    # Run the algorithm
    print(f"\nRunning HSP Lazy Greedy algorithm...")
    start_time = time.time()
    
    selected_elements, final_objective, final_coverage = hsp_lazy_greedy_auto(
        df, views, budget, beta
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Print results
    print(f"\nResults:")
    print(f"  Execution time: {execution_time:.2f} seconds")
    print(f"  Selected elements: {len(selected_elements)}")
    print(f"  Final objective: {final_objective:.4f}")
    print(f"  Views covered: {final_coverage} out of {len(views)}")
    print(f"  Coverage percentage: {(final_coverage / len(views)) * 100:.1f}%")
    
    # Show some selected elements
    print(f"\nFirst 10 selected elements:")
    selected_df = df.iloc[selected_elements[:10]]
    print(selected_df[['id', 'score', 'category', 'region', 'value']].to_string(index=False))
    
    # Analyze selection
    print(f"\nSelection Analysis:")
    selected_df_full = df.iloc[selected_elements]
    print(f"  Average score: {selected_df_full['score'].mean():.2f}")
    print(f"  Average value: {selected_df_full['value'].mean():.2f}")
    print(f"  Category distribution:")
    for cat, count in selected_df_full['category'].value_counts().items():
        print(f"    {cat}: {count} ({count/len(selected_elements)*100:.1f}%)")

if __name__ == "__main__":
    run_example()