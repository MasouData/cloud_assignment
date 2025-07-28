# Replace your existing hsp_lazy_greedy function with this optimized version:

def hsp_lazy_greedy(df, views, budget, beta):
    """
    Optimized version of hsp_lazy_greedy with significant performance improvements.
    
    Key optimizations:
    1. Eliminated redundant cover_mat matrix - use sparse representation
    2. Penalty caching to avoid repeated calculations  
    3. Set-based membership testing for O(1) lookups
    4. Vectorized coverage calculations
    5. More efficient data types
    """
    n, m = len(df), len(views)
    cover_counts = np.zeros(n, dtype=np.int32)
    
    # Build element-to-queries mapping (sparse representation)
    element_to_queries = [[] for _ in range(n)]
    
    for j, view_fun in enumerate(views.values()):
        idx = view_fun(df).index.values  # Convert to numpy array once
        cover_counts[idx] += 1
        for i in idx:
            element_to_queries[i].append(j)
    
    # Convert to numpy arrays for faster vectorized operations
    for i in range(n):
        element_to_queries[i] = np.array(element_to_queries[i], dtype=np.int32)

    # Precompute BIC weights
    bic_weights = 1.0 / (1 + cover_counts)
    
    # Initialize heap
    heap = [(-bic_weights[i], i) for i in range(n)]
    heapq.heapify(heap)

    # Use set for O(1) membership testing instead of boolean array
    selected_set = set()
    selected_list = []
    
    # Track coverage efficiently
    covered_queries = np.zeros(m, dtype=bool)
    
    # Cache penalty calculations since they're expensive
    penalty_cache = {}
    def get_penalty(num_covered):
        if num_covered not in penalty_cache:
            penalty_cache[num_covered] = beta * max(0, num_covered - 1)
        return penalty_cache[num_covered]
    
    current_penalty = 0

    while len(selected_list) < budget and heap:
        neg_bound, i = heapq.heappop(heap)
        
        # O(1) membership test
        if i in selected_set:
            continue

        # Get queries covered by element i
        queries_for_i = element_to_queries[i]
        
        if len(queries_for_i) == 0:
            incremental_coverage = 0
        else:
            # Vectorized computation of incremental coverage
            incremental_coverage = np.sum(~covered_queries[queries_for_i])
        
        if incremental_coverage == 0:
            # No new coverage, penalty doesn't change
            gain = bic_weights[i]
        else:
            new_num_covered = np.sum(covered_queries) + incremental_coverage
            new_penalty = get_penalty(new_num_covered)
            gain = bic_weights[i] - (new_penalty - current_penalty)

        # Lazy evaluation check
        if abs(gain + neg_bound) < 1e-8:
            # Accept this element
            selected_set.add(i)
            selected_list.append(i)
            
            # Update coverage efficiently
            if len(queries_for_i) > 0:
                covered_queries[queries_for_i] = True
            current_penalty = get_penalty(np.sum(covered_queries))
        else:
            # Re-insert with updated gain
            heapq.heappush(heap, (-gain, i))

    # Calculate final objective
    total_bic = np.sum(bic_weights[selected_list])
    final_obj = total_bic - current_penalty
    final_cov_S = np.sum(covered_queries)

    return selected_list, final_obj, final_cov_S