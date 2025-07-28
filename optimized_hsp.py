import numpy as np
import heapq

def hsp_lazy_greedy_optimized(df, views, budget, beta):
    """
    Optimized version of hsp_lazy_greedy with several performance improvements:
    1. Precompute penalty calculations
    2. Use integer operations for coverage tracking
    3. Minimize redundant computations
    4. Early termination optimizations
    """
    n, m = len(df), len(views)
    cover_counts = np.zeros(n, dtype=np.int32)
    cover_mat = np.zeros((n, m), dtype=bool)

    # Build coverage matrix and counts
    for j, view_fun in enumerate(views.values()):
        idx = view_fun(df).index
        cover_counts[idx] += 1
        cover_mat[idx, j] = True

    # Precompute BIC weights
    bic_weights = 1.0 / (1 + cover_counts)
    
    # Initialize heap with all elements
    heap = [(-bic_weights[i], i) for i in range(n)]
    heapq.heapify(heap)

    # Use sets for faster operations
    selected_set = set()
    selected_list = []
    
    # Track coverage more efficiently
    num_covered_queries = 0
    covered_queries = np.zeros(m, dtype=bool)
    
    # Precompute penalty function values to avoid repeated calculations
    penalty_cache = {}
    def get_penalty(num_covered):
        if num_covered not in penalty_cache:
            penalty_cache[num_covered] = beta * max(0, num_covered - 1)
        return penalty_cache[num_covered]
    
    current_penalty = get_penalty(num_covered_queries)

    while len(selected_list) < budget and heap:
        neg_bound, i = heapq.heappop(heap)
        
        # Skip if already selected
        if i in selected_set:
            continue

        # Calculate actual gain more efficiently
        # Find new queries that would be covered by adding element i
        element_queries = cover_mat[i]
        new_queries_mask = element_queries & (~covered_queries)
        incremental_coverage = np.sum(new_queries_mask)
        
        if incremental_coverage == 0:
            # No new coverage, gain is just BIC weight minus any penalty increase
            new_penalty = current_penalty  # No change in penalty
            gain = bic_weights[i]
        else:
            new_num_covered = num_covered_queries + incremental_coverage
            new_penalty = get_penalty(new_num_covered)
            gain = bic_weights[i] - (new_penalty - current_penalty)

        # Check if gain matches bound (lazy evaluation)
        if abs(gain + neg_bound) < 1e-8:
            # Accept this element
            selected_set.add(i)
            selected_list.append(i)
            
            # Update coverage tracking
            covered_queries |= element_queries
            num_covered_queries = np.sum(covered_queries)
            current_penalty = get_penalty(num_covered_queries)
            
        else:
            # Re-insert with updated gain
            heapq.heappush(heap, (-gain, i))

    # Calculate final objective
    total_bic = np.sum(bic_weights[selected_list])
    final_obj = total_bic - current_penalty
    
    return selected_list, final_obj, num_covered_queries


def hsp_lazy_greedy_ultra_optimized(df, views, budget, beta):
    """
    Ultra-optimized version with additional improvements:
    1. Bitwise operations for coverage tracking
    2. Vectorized operations where possible
    3. Memory-efficient data structures
    """
    n, m = len(df), len(views)
    
    # Use more efficient data types
    cover_counts = np.zeros(n, dtype=np.int16)  # Assuming coverage counts are small
    
    # Build coverage matrix more efficiently
    view_indices = []
    for j, view_fun in enumerate(views.values()):
        idx = view_fun(df).index.values  # Convert to numpy array
        view_indices.append(idx)
        cover_counts[idx] += 1
    
    # Create sparse representation of coverage matrix
    element_to_queries = {}
    for i in range(n):
        element_to_queries[i] = []
    
    for j, idx_array in enumerate(view_indices):
        for i in idx_array:
            element_to_queries[i].append(j)
    
    # Convert to numpy arrays for faster access
    for i in range(n):
        element_to_queries[i] = np.array(element_to_queries[i], dtype=np.int16)

    # Precompute BIC weights
    bic_weights = 1.0 / (1 + cover_counts.astype(np.float32))
    
    # Use bitset for covered queries (if m <= 64, otherwise use boolean array)
    if m <= 64:
        covered_queries_bitset = 0
        
        def update_coverage_bitset(element_idx):
            nonlocal covered_queries_bitset
            for q in element_to_queries[element_idx]:
                covered_queries_bitset |= (1 << q)
            return bin(covered_queries_bitset).count('1')
        
        def get_new_coverage_bitset(element_idx):
            new_bitset = covered_queries_bitset
            for q in element_to_queries[element_idx]:
                new_bitset |= (1 << q)
            return bin(new_bitset).count('1') - bin(covered_queries_bitset).count('1')
        
        coverage_update_fn = update_coverage_bitset
        new_coverage_fn = get_new_coverage_bitset
        get_total_coverage = lambda: bin(covered_queries_bitset).count('1')
        
    else:
        covered_queries = np.zeros(m, dtype=bool)
        
        def update_coverage_array(element_idx):
            covered_queries[element_to_queries[element_idx]] = True
            return np.sum(covered_queries)
        
        def get_new_coverage_array(element_idx):
            queries = element_to_queries[element_idx]
            return np.sum(~covered_queries[queries])
        
        coverage_update_fn = update_coverage_array
        new_coverage_fn = get_new_coverage_array
        get_total_coverage = lambda: np.sum(covered_queries)

    # Initialize heap
    heap = [(-bic_weights[i], i) for i in range(n)]
    heapq.heapify(heap)

    selected_set = set()
    selected_list = []
    num_covered_queries = 0
    
    # Penalty calculation
    def penalty(num_covered):
        return beta * max(0, num_covered - 1)
    
    current_penalty = penalty(0)

    while len(selected_list) < budget and heap:
        neg_bound, i = heapq.heappop(heap)
        
        if i in selected_set:
            continue

        # Calculate incremental coverage
        incremental_coverage = new_coverage_fn(i)
        
        if incremental_coverage == 0:
            gain = bic_weights[i]
        else:
            new_num_covered = num_covered_queries + incremental_coverage
            new_penalty = penalty(new_num_covered)
            gain = bic_weights[i] - (new_penalty - current_penalty)

        if abs(gain + neg_bound) < 1e-8:
            # Accept element
            selected_set.add(i)
            selected_list.append(i)
            num_covered_queries = coverage_update_fn(i)
            current_penalty = penalty(num_covered_queries)
        else:
            heapq.heappush(heap, (-gain, i))

    # Final objective
    total_bic = np.sum(bic_weights[selected_list])
    final_obj = total_bic - current_penalty
    
    return selected_list, final_obj, num_covered_queries