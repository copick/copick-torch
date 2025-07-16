"""
parallelization.py - Flexible GPU Processing
==========================================

A thread-safe GPU processing pool that distributes computational tasks across multiple GPUs
with automatic model loading, memory management, and error handling.

For detailed API documentation and examples, see:
https://chanzuckerberg.github.io/saber/api/parallel-inference/
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Any, Dict, Optional
import torch, threading, time
from tqdm import tqdm

class GPUPool:
    """
    Flexible GPU processing pool supporting both threading and multiprocessing.
    
    Usage:
        # Fast threading approach (shared models)
        pool = GPUPool(n_gpus=4, approach="threading")
        
        # Robust multiprocessing approach (isolated models)  
        pool = GPUPool(n_gpus=4, approach="multiprocessing")
        
        # Auto-choose based on model size
        pool = GPUPool(n_gpus=4, approach="auto", model_size_gb=20)
    """
    
    def __init__(self, 
                 init_fn: Optional[Callable] = None,
                 init_args: tuple = (),
                 init_kwargs: dict = {},
                 verbose: bool = True):
        
        self.n_gpus = torch.cuda.device_count()
        self.init_fn = init_fn
        self.init_args = init_args
        self.init_kwargs = init_kwargs
        self.verbose = verbose
        
        # Spawn Threadding Pool
        self._init_threading()
    
    # ============================================================================
    # THREADING IMPLEMENTATION
    # ============================================================================
    
    def _init_threading(self):
        """Initialize threading approach - shared models"""
        self.models = {}  # Shared across threads
        self.model_locks = {}  # Per-GPU locks
        self.initialized = threading.Event()
        
        if self.verbose:
            print(f"GPUPool (threading): {self.n_gpus} GPUs with shared models")
    
    def _initialize_models_threading(self):
        """Load models once, shared across all threads"""
        if self.verbose:
            print("Loading models for threading approach...")
            
        for gpu_id in range(self.n_gpus):
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            
            if self.init_fn:
                if self.verbose:
                    print(f"GPU {gpu_id}: Loading shared models...")
                start_time = time.time()
                models = self.init_fn(gpu_id, *self.init_args, **self.init_kwargs)
                load_time = time.time() - start_time
                
                self.models[gpu_id] = models
                self.model_locks[gpu_id] = threading.RLock()
                
                if self.verbose:
                    gpu_mem = torch.cuda.memory_allocated(gpu_id) / 1e9
                    print(f"GPU {gpu_id}: Models loaded in {load_time:.1f}s, {gpu_mem:.1f}GB VRAM")
            else:
                self.models[gpu_id] = None
                self.model_locks[gpu_id] = threading.RLock()
                
        self.initialized.set()
        if self.verbose:
            print("All shared models ready!\n")
    
    def _execute_threading(self, func, tasks, task_ids, progress_desc):
        """Execute using threading approach"""
        if not self.initialized.is_set():
            self._initialize_models_threading()
            
        def worker_thread(task_data):
            task_id, gpu_id, args, kwargs = task_data
            
            # Wait for initialization
            self.initialized.wait()
            
            # Get exclusive access to this GPU
            with self.model_locks[gpu_id]:
                try:
                    torch.cuda.set_device(gpu_id)
                    
                    # Add GPU context
                    enhanced_kwargs = kwargs.copy()
                    enhanced_kwargs['gpu_id'] = gpu_id
                    if self.models[gpu_id] is not None:
                        enhanced_kwargs['models'] = self.models[gpu_id]
                    
                    start_time = time.time()
                    result = func(*args, **enhanced_kwargs)
                    processing_time = time.time() - start_time
                    
                    return {
                        'success': True,
                        'task_id': task_id,
                        'gpu_id': gpu_id,
                        'processing_time': processing_time,
                        'result': result
                    }
                    
                except Exception as e:
                    return {
                        'success': False,
                        'task_id': task_id,
                        'gpu_id': gpu_id,
                        'error': str(e)
                    }
        
        # Prepare tasks with GPU assignment
        prepared_tasks = []
        for i, (task_id, task) in enumerate(zip(task_ids, tasks)):
            gpu_id = i % self.n_gpus  # Round-robin
            
            if isinstance(task, dict):
                args, kwargs = (), task
            elif isinstance(task, tuple) and len(task) == 2 and isinstance(task[1], dict):
                args, kwargs = task
            elif isinstance(task, (list, tuple)):
                args, kwargs = task, {}
            else:
                args, kwargs = (task,), {}
                
            prepared_tasks.append((task_id, gpu_id, args, kwargs))
        
        # Execute with thread pool
        results = []
        with ThreadPoolExecutor(max_workers=self.n_gpus) as executor:
            with tqdm(total=len(tasks), desc=progress_desc, unit='task', disable=not self.verbose) as pbar:
                future_to_task = {
                    executor.submit(worker_thread, task): task 
                    for task in prepared_tasks
                }
                
                for future in as_completed(future_to_task):
                    result = future.result()
                    results.append(result)
                    
                    if result['success'] and self.verbose:
                        pbar.set_postfix({
                            'GPU': result['gpu_id'],
                            'Task': str(result['task_id'])[:15],
                            'Time': f"{result['processing_time']:.1f}s"
                        })
                    elif not result['success'] and self.verbose:
                        print(f"\n❌ Task {result['task_id']} failed: {result['error']}")
                        
                    if self.verbose:
                        pbar.update(1)
        
        return results
    
    
    # ============================================================================
    # UNIFIED INTERFACE
    # ============================================================================
    
    def execute(self, 
                func: Callable,
                tasks: List[Any],
                task_ids: Optional[List] = None,
                progress_desc: str = "Processing") -> List[Dict]:
        """
        Execute function on all tasks across GPUs.
        
        Your function will receive:
            - All your original arguments
            - gpu_id: int (keyword argument)
            - models: Any (keyword argument, if init_fn was provided)
        """
        if not tasks:
            return []
        
        if task_ids is None:
            task_ids = list(range(len(tasks)))

        results = self._execute_threading(func, tasks, task_ids, progress_desc)
        
        # Print statistics
        if self.verbose:
            self._print_stats(results)
        
        return sorted(results, key=lambda x: x.get('task_id', 0))
    
    def _print_stats(self, results):
        """Print execution statistics"""
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"\n{'='*50}")
        print(f"EXECUTION COMPLETE (THREADING)")
        print(f"{'='*50}")
        print(f"Total tasks: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")

        if failed:
            print(f"Failed runs: {[r['task_id'] for r in failed]}")
            for failed_run in failed:
                print(f"  - {failed_run['task_id']}: {failed_run['error']}")        
        
        if successful:
            gpu_stats = {}
            for r in successful:
                gpu_id = r['gpu_id']
                if gpu_id not in gpu_stats:
                    gpu_stats[gpu_id] = {'count': 0, 'total_time': 0.0}
                gpu_stats[gpu_id]['count'] += 1
                gpu_stats[gpu_id]['total_time'] += r['processing_time']
            
            print(f"\nGPU Statistics:")
            for gpu_id, stats in gpu_stats.items():
                avg_time = stats['total_time'] / stats['count']
                print(f"  GPU {gpu_id}: {stats['count']} tasks, avg {avg_time:.2f}s/task")
    
    def shutdown(self):
        """Shutdown workers"""
        if self.verbose:
            print("Shutting down workers...")
            
            # Send shutdown signals
            for _ in range(self.n_gpus):
                try:
                    self.task_queue.put(None, timeout=1.0)
                except:
                    pass
            
            # self.started = False
            if self.verbose:
                print("All workers shut down")
    
# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def gpu_map(func: Callable, 
            tasks: List[Any],
            approach: str = "auto",
            model_size_gb: float = 1.0,
            n_gpus: Optional[int] = None,
            init_fn: Optional[Callable] = None,
            init_args: tuple = (),
            init_kwargs: dict = {},
            verbose: bool = True) -> List[Dict]:
    """
    Convenience function for GPU mapping with automatic approach selection.
    
    Example:
        # Let it auto-choose based on model size
        results = gpu_map(my_function, tasks, model_size_gb=20)  # Uses multiprocessing
        
        # Force threading for speed
        results = gpu_map(my_function, tasks, approach="threading")
    """
    with GPUPool(n_gpus, approach, init_fn, init_args, init_kwargs, model_size_gb, verbose) as pool:
        return pool.execute(func, tasks)

if __name__ == "__main__":
    print("Flexible GPU Processing Pool")
    print("=" * 40)
    print("Supports both threading (fast) and multiprocessing (robust) approaches")
    
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} CUDA GPUs")
    else:
        print("No CUDA GPUs available")