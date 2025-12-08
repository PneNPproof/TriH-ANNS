/**
 * @file thread_pool_v2.h
 * @brief High-performance thread pool implementation for CPU reranking
 * 
 * This thread pool is specifically designed for the reranking phase of TriH-ANNS,
 * where CPU threads perform precise distance computation using full-dimensional
 * features. The implementation provides:
 * 
 * - Efficient task queuing with minimal synchronization overhead
 * - Automatic thread count detection based on hardware
 * - Exception-safe RAII design with proper cleanup
 * - Future-based result handling for asynchronous operations
 * 
 * The thread pool enables overlapping GPU computation (Phase 1) with CPU
 * reranking (Phase 2) for optimal pipeline efficiency.
 * 
 * Dependencies:
 * - C++17 standard library for std::invoke_result_t
 * - Standard threading library components
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

/**
 * @brief High-performance thread pool for asynchronous task execution
 * 
 * Thread-safe pool that maintains a fixed number of worker threads and
 * executes submitted tasks asynchronously. Designed for compute-intensive
 * workloads where task submission overhead must be minimized.
 */
class ThreadPool {
public:
    /**
     * @brief Constructor - creates and starts worker threads
     * 
     * Initializes the specified number of worker threads that immediately
     * begin waiting for tasks. Each thread runs a continuous loop that
     * waits for tasks and executes them until the pool is destroyed.
     * 
     * @param thread_count Number of worker threads (defaults to hardware concurrency)
     * 
     * @note Using hardware concurrency as default provides good performance
     *       for CPU-bound tasks without oversubscription
     */
    explicit ThreadPool(size_t thread_count = std::thread::hardware_concurrency())
        : stop_(false) {
        workers_.reserve(thread_count);
        for(size_t i = 0; i < thread_count; ++i) {
            workers_.emplace_back([this] {
                while(true) {
                    Task task;
                    {   // Minimize lock scope for better performance
                        std::unique_lock lock(mutex_);
                        cv_.wait(lock, [this] {
                            return stop_ || !tasks_.empty();
                        });

                        // Exit if stopping and no more tasks
                        if(stop_ && tasks_.empty()) return;

                        // Get next task from queue
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }

                    task();  // Execute task outside of lock
                }
            });
        }
    }

    /**
     * @brief Destructor - gracefully shuts down all worker threads
     * 
     * Sets stop flag, notifies all waiting threads, and waits for
     * all worker threads to complete their current tasks and exit.
     * Ensures no tasks are left running when the pool is destroyed.
     */
    ~ThreadPool() {
        {   // Notify all threads to stop
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        
        // Wait for all threads to complete
        for(auto& worker : workers_) {
            if(worker.joinable()) worker.join();
        }
    }

    /**
     * @brief Submit a task for asynchronous execution
     * 
     * Enqueues a callable object with its arguments for execution by
     * worker threads. Returns a future that can be used to retrieve
     * the result or wait for completion.
     * 
     * @tparam F Callable type (function, lambda, functor)
     * @tparam Args Argument types for the callable
     * @param f Callable object to execute
     * @param args Arguments to pass to the callable
     * @return std::future<return_type> Future for the result
     * 
     * @throws std::runtime_error if called on a stopped thread pool
     * 
     * @note Uses perfect forwarding to preserve argument types
     * @note Return type is automatically deduced from callable signature
     * 
     * Example usage:
     * @code
     * auto future = pool.enqueue([](int x, int y) { return x + y; }, 3, 4);
     * int result = future.get();  // result = 7
     * @endcode
     */
    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<std::invoke_result_t<F, Args...>> {
        
        using return_type = std::invoke_result_t<F, Args...>;
        
        // Create packaged task for the callable and its arguments
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> res = task->get_future();
        {   // Add task to queue under lock
            std::lock_guard lock(mutex_);
            if(stop_) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks_.emplace([task](){ (*task)(); });
        }
        cv_.notify_one();  // Wake up one waiting worker
        return res;
    }

private:
    using Task = std::function<void()>;  ///< Type-erased task wrapper
    
    std::vector<std::thread> workers_;   ///< Worker thread pool
    std::queue<Task> tasks_;             ///< Task queue (FIFO)
    
    // Synchronization primitives
    std::mutex mutex_;                   ///< Protects task queue and stop flag
    std::condition_variable cv_;         ///< Signals workers when tasks available
    
    // Control flags
    std::atomic<bool> stop_;             ///< Signal for graceful shutdown
};
