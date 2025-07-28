import numpy as np
import heapq

def hsp_lazy_greedy(df, views, budget, beta):
    """
    Optimized version of hsp_lazy_greedy with significant performance improvements:
    
    Key optimizations:
    1. Penalty caching to avoid repeated calculations
    2. Efficient coverage tracking using integer counters
    3. Set-based membership testing for selected elements
    4. Reduced numpy array operations
    5. Early exit conditions
    """
    n, m = len(df), len(views)
    cover_counts = np.zeros(n, dtype=np.int32)
    
    # Build coverage information more efficiently
    view_indices = []
    for j, view_fun in enumerate(views.values()):
        idx = view_fun(df).index.values
        view_indices.append(idx)
        cover_counts[idx] += 1
    
    # Create element-to-queries mapping for faster lookup
    element_queries = [[] for _ in range(n)]
    for j, idx_array in enumerate(view_indices):
        for i in idx_array:
            element_queries[i].append(j)
    
    # Convert to numpy arrays for faster access
    for i in range(n):
        element_queries[i] = np.array(element_queries[i], dtype=np.int16)

    # Precompute BIC weights
    bic_weights = 1.0 / (1 + cover_counts)
    
    # Initialize heap
    heap = [(-bic_weights[i], i) for i in range(n)]
    heapq.heapify(heap)

    # Use set for O(1) membership testing
    selected_set = set()
    selected_list = []
    
    # Track coverage efficiently
    covered_queries = np.zeros(m, dtype=bool)
    num_covered_queries = 0
    
    # Cache penalty calculations
    penalty_cache = {}
    def get_penalty(num_covered):
        if num_covered not in penalty_cache:
            penalty_cache[num_covered] = beta * max(0, num_covered - 1)
        return penalty_cache[num_covered]
    
    current_penalty = 0

    while len(selected_list) < budget and heap:
        neg_bound, i = heapq.heappop(heap)
        
        # Skip if already selected
        if i in selected_set:
            continue

        # Calculate incremental coverage efficiently
        queries_for_i = element_queries[i]
        if len(queries_for_i) == 0:
            incremental_coverage = 0
        else:
            incremental_coverage = np.sum(~covered_queries[queries_for_i])
        
        if incremental_coverage == 0:
            # No new coverage - just BIC weight
            gain = bic_weights[i]
        else:
            new_num_covered = num_covered_queries + incremental_coverage
            new_penalty = get_penalty(new_num_covered)
            gain = bic_weights[i] - (new_penalty - current_penalty)

        # Lazy evaluation check
        if abs(gain + neg_bound) < 1e-8:
            # Accept this element
            selected_set.add(i)
            selected_list.append(i)
            
            # Update coverage
            if len(queries_for_i) > 0:
                covered_queries[queries_for_i] = True
                num_covered_queries = np.sum(covered_queries)
            current_penalty = get_penalty(num_covered_queries)
        else:
            # Re-insert with updated gain
            heapq.heappush(heap, (-gain, i))

    # Calculate final objective
    total_bic = np.sum(bic_weights[selected_list])
    final_obj = total_bic - current_penalty
    final_cov_S = num_covered_queries

    return selected_list, final_obj, final_cov_S