import numpy as np
import pandas as pd
import time
from hsp_optimized import hsp_lazy_greedy, hsp_lazy_greedy_optimized, hsp_lazy_greedy_auto

def create_synthetic_data(n, m, sparsity=0.1):
    """
    Create synthetic data for benchmarking
    
    Args:
        n: Number of elements (rows in dataframe)
        m: Number of views (queries)
        sparsity: Fraction of elements covered by each view
    """
    # Create a simple dataframe with some features
    df = pd.DataFrame({
        'id': range(n),
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n)
    })
    
    # Create views that select different subsets of the data
    views = {}
    
    for i in range(m):
        # Each view covers a random subset of elements
        if i < m // 4:
            # Feature-based views
            threshold = np.random.uniform(-1, 1)
            views[f'view_feature1_{i}'] = lambda df, t=threshold: df[df['feature1'] > t]
        elif i < m // 2:
            # Category-based views
            cat = np.random.choice(['A', 'B', 'C', 'D'])
            views[f'view_category_{i}'] = lambda df, c=cat: df[df['category'] == c]
        else:
            # Random subset views
            subset_size = int(n * sparsity)
            indices = np.random.choice(n, subset_size, replace=False)
            views[f'view_random_{i}'] = lambda df, idx=indices: df.iloc[idx]
    
    return df, views

def benchmark_implementations(n_values, m=50, budget=100, beta=0.1):
    """
    Benchmark different implementations across various problem sizes
    """
    results = []
    
    for n in n_values:
        print(f"\nBenchmarking n={n:,}")
        
        # Create synthetic data
        df, views = create_synthetic_data(n, m)
        
        # Test original implementation (skip for very large n)
        if n <= 1000000:
            start_time = time.time()
            try:
                result_orig = hsp_lazy_greedy(df, views, budget, beta)
                time_orig = time.time() - start_time
                print(f"  Original: {time_orig:.2f}s")
            except MemoryError:
                time_orig = float('inf')
                print(f"  Original: Memory Error")
        else:
            time_orig = float('inf')
            print(f"  Original: Skipped (too large)")
        
        # Test optimized implementation
        start_time = time.time()
        try:
            result_opt = hsp_lazy_greedy_optimized(df, views, budget, beta)
            time_opt = time.time() - start_time
            print(f"  Optimized: {time_opt:.2f}s")
        except Exception as e:
            time_opt = float('inf')
            print(f"  Optimized: Error - {e}")
        
        # Test auto implementation
        start_time = time.time()
        try:
            result_auto = hsp_lazy_greedy_auto(df, views, budget, beta)
            time_auto = time.time() - start_time
            print(f"  Auto: {time_auto:.2f}s")
        except Exception as e:
            time_auto = float('inf')
            print(f"  Auto: Error - {e}")
        
        # Calculate speedup
        if time_orig != float('inf') and time_opt != float('inf'):
            speedup = time_orig / time_opt
            print(f"  Speedup: {speedup:.1f}x")
        
        results.append({
            'n': n,
            'time_original': time_orig,
            'time_optimized': time_opt,
            'time_auto': time_auto,
            'speedup': time_orig / time_opt if time_orig != float('inf') and time_opt != float('inf') else None
        })
    
    return results

def memory_usage_test():
    """
    Test memory usage patterns
    """
    print("\nMemory Usage Analysis:")
    
    import psutil
    import os
    
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    n_values = [100000, 1000000, 5000000]
    
    for n in n_values:
        print(f"\nTesting n={n:,}")
        
        initial_memory = get_memory_usage()
        
        # Create data
        df, views = create_synthetic_data(n, 20)
        after_data_memory = get_memory_usage()
        
        # Run optimized version
        result = hsp_lazy_greedy_optimized(df, views, 50, 0.1)
        after_algorithm_memory = get_memory_usage()
        
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  After data creation: {after_data_memory:.1f} MB (+{after_data_memory - initial_memory:.1f} MB)")
        print(f"  After algorithm: {after_algorithm_memory:.1f} MB (+{after_algorithm_memory - after_data_memory:.1f} MB)")
        print(f"  Total memory increase: {after_algorithm_memory - initial_memory:.1f} MB")

if __name__ == "__main__":
    print("HSP Lazy Greedy Optimization Benchmark")
    print("=" * 50)
    
    # Test different problem sizes
    n_values = [10000, 100000, 1000000, 5000000, 10000000]
    
    # Run benchmarks
    results = benchmark_implementations(n_values)
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    
    for result in results:
        n = result['n']
        speedup = result['speedup']
        print(f"n={n:>8,}: ", end="")
        
        if speedup is not None:
            print(f"{speedup:>5.1f}x speedup")
        else:
            print("Original too slow/failed")
    
    # Memory usage test
    try:
        memory_usage_test()
    except ImportError:
        print("\npsutil not available for memory testing")
    
    print("\nRecommendations:")
    print("- For n < 1M: Use original implementation")
    print("- For 1M ≤ n < 50M: Use hsp_lazy_greedy_optimized")
    print("- For n ≥ 50M: Use hsp_lazy_greedy_memory_optimized")
    print("- For automatic selection: Use hsp_lazy_greedy_auto")