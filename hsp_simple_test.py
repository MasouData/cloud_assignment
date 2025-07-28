#!/usr/bin/env python3
"""
Simple test script for HSP optimization that works with just numpy and pandas
"""

import numpy as np
import pandas as pd
import time
import heapq

def hsp_lazy_greedy_basic_optimized(df, views, budget, beta):
    """
    Basic optimized version that works without numba but still provides significant speedups
    for large n through:
    1. Memory-efficient data structures
    2. Optimized heap operations
    3. Sparse element_queries representation
    4. Early termination strategies
    """
    n, m = len(df), len(views)
    
    # Use memory-efficient data types
    cover_counts = np.zeros(n, dtype=np.uint16)
    
    # Build coverage information
    view_indices = []
    for j, view_fun in enumerate(views.values()):
        idx = view_fun(df).index.values.astype(np.int32)
        view_indices.append(idx)
        cover_counts[idx] += 1
    
    # Use sparse representation for element_queries to save memory
    element_queries_dict = {}
    
    for j, idx_array in enumerate(view_indices):
        for i in idx_array:
            if i not in element_queries_dict:
                element_queries_dict[i] = []
            element_queries_dict[i].append(j)
    
    # Convert to numpy arrays only for elements that have queries
    element_queries = {}
    for i, queries in element_queries_dict.items():
        element_queries[i] = np.array(queries, dtype=np.int32)
    
    # Precompute BIC weights with overflow protection
    bic_weights = np.divide(1.0, (1 + cover_counts.astype(np.float32)), 
                           out=np.zeros_like(cover_counts, dtype=np.float32), 
                           where=(cover_counts > 0))
    
    # Initialize heap more efficiently - only include elements with non-zero weights
    heap = []
    for i in range(n):
        if bic_weights[i] > 0:
            heap.append((-bic_weights[i], i))
    
    heapq.heapify(heap)

    # Use boolean array for large n (more memory efficient than set)
    selected_set = np.zeros(n, dtype=bool)
    selected_list = []
    
    # Track coverage efficiently
    covered_queries = np.zeros(m, dtype=bool)
    num_covered_queries = 0
    
    # Optimized penalty calculation with memoization
    penalty_cache = {}
    def get_penalty(num_covered):
        if num_covered not in penalty_cache:
            penalty_cache[num_covered] = beta * max(0, num_covered - 1)
        return penalty_cache[num_covered]
    
    current_penalty = 0
    
    # Add progress tracking
    iteration_count = 0
    max_iterations = min(budget * 10, n)

    while len(selected_list) < budget and heap and iteration_count < max_iterations:
        iteration_count += 1
        
        neg_bound, i = heapq.heappop(heap)
        
        # Skip if already selected
        if selected_set[i]:
            continue

        # Calculate incremental coverage
        if i in element_queries:
            queries_for_i = element_queries[i]
            incremental_coverage = np.sum(~covered_queries[queries_for_i])
        else:
            incremental_coverage = 0
        
        if incremental_coverage == 0:
            gain = bic_weights[i]
        else:
            new_num_covered = num_covered_queries + incremental_coverage
            new_penalty = get_penalty(new_num_covered)
            gain = bic_weights[i] - (new_penalty - current_penalty)

        # Lazy evaluation check
        tolerance = 1e-10 if n > 1000000 else 1e-8
        if abs(gain + neg_bound) < tolerance:
            # Accept this element
            selected_set[i] = True
            selected_list.append(i)
            
            # Update coverage
            if i in element_queries:
                queries_for_i = element_queries[i]
                covered_queries[queries_for_i] = True
                num_covered_queries = np.sum(covered_queries)
            
            current_penalty = get_penalty(num_covered_queries)
            
            # Clear penalty cache periodically to manage memory
            if len(penalty_cache) > 1000:
                penalty_cache.clear()
                penalty_cache[num_covered_queries] = current_penalty
                
        else:
            # Re-insert with updated gain
            heapq.heappush(heap, (-gain, i))

    # Calculate final objective
    selected_indices = np.array(selected_list, dtype=np.int32)
    total_bic = np.sum(bic_weights[selected_indices])
    final_obj = total_bic - current_penalty
    final_cov_S = num_covered_queries

    return selected_list, final_obj, final_cov_S


def create_test_data(n=100000, m=50):
    """Create test data for benchmarking"""
    print(f"Creating test dataset with n={n:,} elements and m={m} views...")
    
    # Create a dataframe
    np.random.seed(42)
    df = pd.DataFrame({
        'id': range(n),
        'score': np.random.uniform(0, 100, n),
        'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'value': np.random.exponential(10, n)
    })
    
    # Create views
    views = {}
    sparsity = 0.1  # Each view covers ~10% of elements
    
    for i in range(m):
        if i % 5 == 0:
            # Score-based views
            threshold = np.random.uniform(20, 80)
            views[f'score_gt_{i}'] = lambda df, t=threshold: df[df['score'] > t]
        elif i % 5 == 1:
            # Category-based views
            cat = np.random.choice(['A', 'B', 'C', 'D', 'E'])
            views[f'category_{cat}_{i}'] = lambda df, c=cat: df[df['category'] == c]
        elif i % 5 == 2:
            # Region-based views
            reg = np.random.choice(['North', 'South', 'East', 'West'])
            views[f'region_{reg}_{i}'] = lambda df, r=reg: df[df['region'] == r]
        elif i % 5 == 3:
            # Value-based views
            threshold = np.random.uniform(5, 15)
            views[f'value_lt_{i}'] = lambda df, t=threshold: df[df['value'] < t]
        else:
            # Random subset views
            indices = np.random.choice(n, size=int(n * sparsity), replace=False)
            views[f'random_{i}'] = lambda df, idx=indices: df.iloc[idx]
    
    return df, views


def run_performance_test():
    """Run a performance test to demonstrate the optimizations"""
    
    # Test with different sizes
    test_sizes = [10000, 100000, 500000]
    budget = 100
    beta = 0.1
    
    for n in test_sizes:
        print(f"\n{'='*60}")
        print(f"Testing with n={n:,} elements")
        print(f"{'='*60}")
        
        # Create test data
        df, views = create_test_data(n=n, m=20)
        
        print(f"Dataset created: {len(df):,} elements, {len(views)} views")
        print(f"Budget: {budget}, Beta: {beta}")
        
        # Run optimized version
        start_time = time.time()
        selected, obj_val, coverage = hsp_lazy_greedy_basic_optimized(df, views, budget, beta)
        end_time = time.time()
        
        print(f"\nResults:")
        print(f"  Selected elements: {len(selected)}")
        print(f"  Objective value: {obj_val:.6f}")
        print(f"  Coverage: {coverage}")
        print(f"  Runtime: {end_time - start_time:.3f} seconds")
        print(f"  Elements per second: {n / (end_time - start_time):,.0f}")
        
        # Memory usage estimation
        memory_mb = (n * 4 + len(views) * n * 0.1 * 4) / (1024 * 1024)
        print(f"  Estimated memory usage: {memory_mb:.1f} MB")


if __name__ == "__main__":
    print("HSP Lazy Greedy Algorithm - Basic Optimization Test")
    print("=" * 60)
    
    try:
        run_performance_test()
        print(f"\n{'='*60}")
        print("Test completed successfully!")
        print("\nKey optimizations demonstrated:")
        print("1. Memory-efficient data structures (uint16, sparse storage)")
        print("2. Optimized heap operations (only non-zero weights)")
        print("3. Boolean arrays instead of sets for large n")
        print("4. Penalty calculation caching")
        print("5. Early termination strategies")
        print("\nFor even better performance with n >= 10M, install numba:")
        print("  pip install numba")
        print("  Then use hsp_lazy_greedy_optimized() from hsp_optimized.py")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()