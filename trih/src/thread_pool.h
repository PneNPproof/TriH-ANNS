/**
 * @file thread_pool.h
 * @brief Legacy thread pool implementation (C++11 compatible)
 * 
 * This header provides an alternative thread pool implementation that is
 * compatible with older C++ standards (C++11). While the system primarily
 * uses thread_pool_v2.h which requires C++17, this version may be used
 * for compatibility with older compilers or build environments.
 * 
 * Key differences from thread_pool_v2.h:
 * - Uses std::result_of instead of std::invoke_result_t (C++11 compatible)
 * - May have slightly different performance characteristics
 * - Maintains compatibility with older C++ standard libraries
 * 
 * Dependencies:
 * - C++11 standard library components
 * - Standard threading library
 */

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

/**
 * @brief C++11 compatible thread pool implementation
 * 
 * Legacy thread pool class that provides similar functionality to
 * thread_pool_v2.h but with C++11 compatibility. Maintains a fixed
 * number of worker threads for asynchronous task execution.
 */
class ThreadPool {
public:
    /**
     * @brief Constructor - creates specified number of worker threads
     * 
     * @param threads Number of worker threads to create
     */
    ThreadPool(size_t);
    
    /**
     * @brief Submit task for asynchronous execution
     * 
     * C++11 compatible version using std::result_of for type deduction.
     * 
     * @tparam F Callable type
     * @tparam Args Argument types
     * @param f Callable to execute
     * @param args Arguments for the callable
     * @return std::future for the result
     */
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    
    /**
     * @brief Destructor - stops all threads and waits for completion
     */
    ~ThreadPool();
    
private:
    // Worker thread management
    std::vector< std::thread > workers;        ///< Pool of worker threads
    
    // Task queue
    std::queue< std::function<void()> > tasks; ///< FIFO task queue
    
    // Synchronization primitives
    std::mutex queue_mutex;                    ///< Protects task queue
    std::condition_variable condition;         ///< Signals task availability
    bool stop;                                 ///< Shutdown flag
};
 
// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads)
    :   stop(false)
{
    for(size_t i = 0;i<threads;++i)
        workers.emplace_back(
            [this]
            {
                for(;;)
                {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock,
                            [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    task();
                }
            }
        );
}

// add new work item to the pool
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // don't allow enqueueing after stopping the pool
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for(std::thread &worker: workers)
        worker.join();
}

#endif