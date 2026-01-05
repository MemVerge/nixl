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
#ifndef GISMO_SEMAPHORE_H
#define GISMO_SEMAPHORE_H

#include <semaphore.h>

class gismoSemaphore {
public:
    gismoSemaphore(size_t count) : count_(count) {
        sem_ = new sem_t;
        sem_init(sem_, 0, 0);
    };

    ~gismoSemaphore() {
        sem_destroy(sem_);
        delete sem_;
    };

    void
    signal() {
        sem_post(sem_);
    };

    void
    wait() {
        sem_wait(sem_);
    };

    void
    waitAll() {
        for (size_t i = 0; i < count_; ++i) {
            wait();
        }
    }

private:
    size_t count_;
    sem_t *sem_;
};


#endif // GISMO_SEMAPHORE_H
