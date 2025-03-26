#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
public:
    explicit ThreadPool(size_t thread_count = std::thread::hardware_concurrency())
        : stop_(false) {
        workers_.reserve(thread_count);
        for(size_t i = 0; i < thread_count; ++i) {
            workers_.emplace_back([this] {
                while(true) {
                    Task task;
                    {   // 减小锁作用域范围
                        std::unique_lock lock(mutex_);
                        cv_.wait(lock, [this] {
                            return stop_ || !tasks_.empty();
                        });

                        if(stop_ && tasks_.empty()) return;

                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }

                    task();  // 执行任务
                }
            });
        }
    }

    ~ThreadPool() {
        {   // 通知所有线程停止
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        
        // 等待所有线程完成
        for(auto& worker : workers_) {
            if(worker.joinable()) worker.join();
        }
    }

    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<std::invoke_result_t<F, Args...>> {
        
        using return_type = std::invoke_result_t<F, Args...>;
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> res = task->get_future();
        {   // 添加任务到队列
            std::lock_guard lock(mutex_);
            if(stop_) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks_.emplace([task](){ (*task)(); });
        }
        cv_.notify_one();
        return res;
    }

private:
    using Task = std::function<void()>;
    
    std::vector<std::thread> workers_;
    std::queue<Task> tasks_;
    
    // 同步原语
    std::mutex mutex_;
    std::condition_variable cv_;
    
    // 停止标志
    std::atomic<bool> stop_;
};
