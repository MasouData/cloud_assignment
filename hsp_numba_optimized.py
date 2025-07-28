import numpy as np
import heapq
from numba import jit, types
from numba.typed import List

@jit(nopython=True)
def compute_incremental_coverage_numba(queries_for_i, covered_queries):
    """Numba-optimized function to compute incremental coverage."""
    count = 0
    for q in queries_for_i:
        if not covered_queries[q]:
            count += 1
    return count

@jit(nopython=True)
def update_coverage_numba(queries_for_i, covered_queries):
    """Numba-optimized function to update coverage."""
    for q in queries_for_i:
        covered_queries[q] = True
    return np.sum(covered_queries)

def hsp_lazy_greedy_numba(df, views, budget, beta):
    """
    Numba-optimized version of hsp_lazy_greedy for maximum performance.
    
    This version uses JIT compilation for the most computationally intensive parts.
    """
    n, m = len(df), len(views)
    cover_counts = np.zeros(n, dtype=np.int32)
    
    # Build coverage information
    view_indices = []
    for j, view_fun in enumerate(views.values()):
        idx = view_fun(df).index.values
        view_indices.append(idx)
        cover_counts[idx] += 1
    
    # Create element-to-queries mapping
    element_queries = []
    for i in range(n):
        queries = []
        for j, idx_array in enumerate(view_indices):
            if i in idx_array:
                queries.append(j)
        element_queries.append(np.array(queries, dtype=np.int32))

    # Precompute BIC weights
    bic_weights = 1.0 / (1 + cover_counts.astype(np.float64))
    
    # Initialize heap
    heap = [(-bic_weights[i], i) for i in range(n)]
    heapq.heapify(heap)

    selected_set = set()
    selected_list = []
    covered_queries = np.zeros(m, dtype=np.bool_)
    num_covered_queries = 0
    
    # Penalty calculation with caching
    penalty_cache = {}
    def get_penalty(num_covered):
        if num_covered not in penalty_cache:
            penalty_cache[num_covered] = beta * max(0, num_covered - 1)
        return penalty_cache[num_covered]
    
    current_penalty = 0

    while len(selected_list) < budget and heap:
        neg_bound, i = heapq.heappop(heap)
        
        if i in selected_set:
            continue

        queries_for_i = element_queries[i]
        
        if len(queries_for_i) == 0:
            incremental_coverage = 0
        else:
            incremental_coverage = compute_incremental_coverage_numba(queries_for_i, covered_queries)
        
        if incremental_coverage == 0:
            gain = bic_weights[i]
        else:
            new_num_covered = num_covered_queries + incremental_coverage
            new_penalty = get_penalty(new_num_covered)
            gain = bic_weights[i] - (new_penalty - current_penalty)

        if abs(gain + neg_bound) < 1e-8:
            selected_set.add(i)
            selected_list.append(i)
            
            if len(queries_for_i) > 0:
                num_covered_queries = update_coverage_numba(queries_for_i, covered_queries)
            current_penalty = get_penalty(num_covered_queries)
        else:
            heapq.heappush(heap, (-gain, i))

    total_bic = np.sum(bic_weights[selected_list])
    final_obj = total_bic - current_penalty
    final_cov_S = num_covered_queries

    return selected_list, final_obj, final_cov_S


def hsp_lazy_greedy_fast(df, views, budget, beta):
    """
    Fast version without numba dependency - should work as drop-in replacement.
    
    This version focuses on algorithmic optimizations rather than JIT compilation.
    """
    n, m = len(df), len(views)
    
    # Pre-allocate arrays with appropriate dtypes
    cover_counts = np.zeros(n, dtype=np.int32)
    
    # Build coverage mapping more efficiently
    element_to_queries = [[] for _ in range(n)]
    
    for j, view_fun in enumerate(views.values()):
        idx = view_fun(df).index.values
        cover_counts[idx] += 1
        for i in idx:
            element_to_queries[i].append(j)
    
    # Convert to numpy arrays for vectorized operations
    for i in range(n):
        element_to_queries[i] = np.array(element_to_queries[i], dtype=np.int32)

    # Precompute weights
    bic_weights = np.reciprocal(1.0 + cover_counts.astype(np.float64))
    
    # Use priority queue with lazy updates
    heap = [(-bic_weights[i], i) for i in range(n)]
    heapq.heapify(heap)

    selected = set()
    selected_list = []
    covered_queries = np.zeros(m, dtype=bool)
    
    # Precompute penalty values for likely range
    max_penalty_range = min(m, 100)  # Reasonable upper bound
    penalty_lookup = np.array([beta * max(0, i - 1) for i in range(max_penalty_range + 1)])
    
    def fast_penalty(num_covered):
        if num_covered <= max_penalty_range:
            return penalty_lookup[num_covered]
        return beta * (num_covered - 1)
    
    current_penalty = 0.0

    while len(selected_list) < budget and heap:
        neg_bound, i = heapq.heappop(heap)
        
        if i in selected:
            continue

        queries = element_to_queries[i]
        
        if len(queries) == 0:
            incremental_coverage = 0
        else:
            # Vectorized computation of new coverage
            incremental_coverage = np.sum(~covered_queries[queries])
        
        if incremental_coverage == 0:
            gain = bic_weights[i]
        else:
            new_num_covered = np.sum(covered_queries) + incremental_coverage
            new_penalty = fast_penalty(new_num_covered)
            gain = bic_weights[i] - (new_penalty - current_penalty)

        if abs(gain + neg_bound) < 1e-8:
            selected.add(i)
            selected_list.append(i)
            
            if len(queries) > 0:
                covered_queries[queries] = True
            current_penalty = fast_penalty(np.sum(covered_queries))
        else:
            heapq.heappush(heap, (-gain, i))

    total_bic = np.sum(bic_weights[selected_list])
    final_obj = total_bic - current_penalty
    final_cov_S = np.sum(covered_queries)

    return selected_list, final_obj, final_cov_S