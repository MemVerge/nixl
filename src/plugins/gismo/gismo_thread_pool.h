/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __GISMO_THREAD_POOL_H
#define __GISMO_THREAD_POOL_H

#include <type_traits>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <iostream>
#include <cassert>

#include "gismo_concurrent_map.h"
#include "gismo_blocking_list.h"

#define THREAD_NAME_LEN 16
#define THREAD_IDLE_PERIOD 5000

class gismoThreadPool {
public:
    gismoThreadPool(const char *pool, size_t size);
    template<class F, class... Args>
    auto
    enqueue(F &&f, Args &&...args) -> std::future<typename std::invoke_result<F, Args...>::type>;
    ~gismoThreadPool();
    void
    addDaemonWorker(std::thread &nt, const char *name);

    bool
    stopping() {
        return requireStop_.load();
    };

    int
    getStaticWorkerCount() {
        return workers_.size();
    }

    bool
    canCreateNewThread() {
        return (dynamicWorkers_.size() + workers_.size()) < numLogicalProcessors_;
    }

private:
    // fixed numbre of threads in pool
    std::vector<std::thread> workers_;
    // task queue
    gismoBlockingList<std::function<void()>> tasks_;

    // synchronization
    std::mutex queueMtx_;
    std::condition_variable condition_;
    std::atomic_bool requireStop_;
    // dynamic threads
    std::shared_mutex dynamicWorkerMtx_;
    std::list<std::thread> dynamicWorkers_;
    gismoConcurrentMap<std::thread::id, bool> dynamicWorkerMap_;
    std::string poolName_;
    std::atomic<int> dynamicThreadId_;
    std::atomic<int> idleThreads_;

    unsigned int numLogicalProcessors_ = std::thread::hardware_concurrency();

private:
    void
    addWorkerIfNeeded();
    bool
    needAddWorker(bool check_threads);
    void
    cleanupDynamicWorkers();

    bool
    processTask();

    void
    normalWorker();

    void
    monitorWorker();

    void
    dynamicWorker();

    void
    setThreadName(std::thread *thread, const char *name, int tid);
};

// return whether we have processed task
inline bool
gismoThreadPool::processTask() {
    auto task = tasks_.pop_front();
    if (task.has_value()) {
        idleThreads_.fetch_sub(1);
        task.value()();
        idleThreads_.fetch_add(1);
    }
    return task.has_value();
}

inline void
gismoThreadPool::normalWorker() {
    for (;;) {
        // try to get one task to execute
        if (processTask()) {
            continue;
        }
        // when no task, wait for condition
        std::unique_lock<std::mutex> lock(queueMtx_);
        condition_.wait(lock, [this] { return requireStop_.load() || !tasks_.empty(); });
        if (requireStop_.load() && tasks_.empty()) return;
    }
}

inline void
gismoThreadPool::monitorWorker() {
    for (; !requireStop_.load();) {
        addWorkerIfNeeded();
        if (tasks_.empty()) {
            cleanupDynamicWorkers();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

inline void
gismoThreadPool::dynamicWorker() {
    auto tid = std::this_thread::get_id();
    idleThreads_.fetch_add(1);
    dynamicWorkerMap_.put(tid, false);
    int idled = 0;
    for (; !requireStop_.load();) {
        if (processTask()) {
            idled = 0;
            continue;
        }
        if (idled > THREAD_IDLE_PERIOD) {
            dynamicWorkerMap_.put(tid, true);
            idleThreads_.fetch_sub(1);
            return;
        } else {
            // this means queue is empty, let's wait until timeout
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            ++idled;
        }
    }
}

// launches fixed amount of workers
inline gismoThreadPool::gismoThreadPool(const char *pool, size_t threads)
    : requireStop_(false),
      poolName_(pool),
      dynamicThreadId_(0),
      idleThreads_(threads) {
    if (poolName_.size() > THREAD_NAME_LEN - 4) {
        poolName_ = poolName_.substr(0, THREAD_NAME_LEN - 4);
    }
    // add worker threads
    for (size_t i = 0; i < threads; ++i) {
        workers_.emplace_back(&gismoThreadPool::normalWorker, this);
        setThreadName(&workers_[i], poolName_.c_str(), i);
    }
    // add monitor thread
    workers_.emplace_back(&gismoThreadPool::monitorWorker, this);
    setThreadName(&workers_.back(), (poolName_ + "-mon").c_str(), -1);
}

void inline gismoThreadPool::addDaemonWorker(std::thread &t, const char *name) {
    workers_.emplace_back(std::move(t));
    setThreadName(&workers_.back(), name, -1);
}

// add new work item to the pool
template<class F, class... Args>
auto
gismoThreadPool::enqueue(F &&f, Args &&...args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::invoke_result<F, Args...>::type;
    // don't allow enqueueing after stopping the pool
    if (requireStop_.load()) throw std::runtime_error("enqueue on stopped ThreadPool");

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    auto res = task->get_future();
    tasks_.push_back([task]() { (*task)(); });

    condition_.notify_one();
    return res;
}

inline bool
gismoThreadPool::needAddWorker(bool check_threads) {
    if (idleThreads_.load() > 0) {
        return false;
    }
    // when there are too many dynamic threads, do not add more
    if (check_threads && dynamicWorkers_.size() - dynamicWorkerMap_.countIf([](bool &v) {
            return v;
        }) > workers_.size() * 8) {
        return false;
    }
    return (tasks_.size() > workers_.size() / 2);
}

inline void
gismoThreadPool::setThreadName(std::thread *thread, const char *name, int tid) {
    char t_name[THREAD_NAME_LEN] = {0};
    if (tid >= 0) {
        snprintf(t_name, THREAD_NAME_LEN - 1, "%s-%X", name, tid);
    } else {
        snprintf(t_name, THREAD_NAME_LEN - 1, "%s", name);
    }
    auto ret = pthread_setname_np(thread->native_handle(), t_name);
    if (ret) {
        std::cerr << "Failed to set " << t_name << ", err " << ret << std::endl;
    }
}

inline void
gismoThreadPool::addWorkerIfNeeded() {
    if (!needAddWorker(false)) {
        return;
    }
    if (!dynamicWorkerMtx_.try_lock()) {
        return;
    }
    if (!needAddWorker(true)) {
        dynamicWorkerMtx_.unlock();
        return;
    }
    // add dynamic workers to make sure there is no tasks in queue
    dynamicWorkers_.emplace_back(&gismoThreadPool::dynamicWorker, this);
    auto name = poolName_ + "-d";
    setThreadName(&dynamicWorkers_.back(), name.c_str(), dynamicThreadId_.fetch_add(1));
    dynamicWorkerMtx_.unlock();
}

inline void
gismoThreadPool::cleanupDynamicWorkers() {
    if (!dynamicWorkerMtx_.try_lock()) {
        return;
    }

    for (auto itr = dynamicWorkers_.begin(); itr != dynamicWorkers_.end();) {
        auto tid = itr->get_id();
        auto val = dynamicWorkerMap_.get(tid);
        if (val.has_value() && val.value()) {
            if (itr->joinable()) {
                itr->join();
            }
            itr = dynamicWorkers_.erase(itr);
            dynamicWorkerMap_.remove(tid);
        } else {
            ++itr;
        }
    }
    dynamicWorkerMtx_.unlock();
}

// the destructor joins all threads
inline gismoThreadPool::~gismoThreadPool() {
    requireStop_.store(true);
    condition_.notify_all();
    for (auto &worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    for (auto &worker : dynamicWorkers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

#endif //__GISMO_THREAD_POOL_H