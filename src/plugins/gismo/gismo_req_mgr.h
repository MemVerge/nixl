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

#ifndef __GISMO_REQ_MGR_H
#define __GISMO_REQ_MGR_H

#include <cstdint>
#include <cassert>

#include "gismo_concurrent_map.h"

struct descMeta {
    int submitCount = 0;
    int doneCount = 0;
    uint64_t updateTime = 0;
};

class gismoReqMgr {
public:
    inline gismoReqMgr() {};
    inline ~gismoReqMgr() {};

    inline void
    markDescDone(uintptr_t req_handle, uintptr_t desc_handle) {
        auto mm = reqs_.get(req_handle);
        assert(mm);
        mm.value()->update(desc_handle, [](descMeta &meta) { meta.doneCount++; });
    }

    inline void
    transferDesc(uintptr_t req_handle, uintptr_t desc_handle) {
        auto mm = reqs_.get(req_handle);
        assert(mm);
        mm.value()->update(desc_handle, [](descMeta &meta) { meta.submitCount++; });
    }

    inline void
    addReq(uintptr_t req_handle) {
        auto mm = new gismoConcurrentMap<uintptr_t, descMeta>();
        reqs_.put(req_handle, mm);
    }

    inline bool
    containsReq(uintptr_t req_handle) {
        return reqs_.contains(req_handle);
    }

    inline bool
    isDescDone(uintptr_t req_handle, uintptr_t desc_handle) {
        auto mm = reqs_.get(req_handle);
        if (!mm) {
            return true;
        }
        auto v = mm.value()->get(desc_handle);
        if (!v) {
            return true;
        }
        return v.value().doneCount == v.value().submitCount;
    }

    inline void
    removeReq(uintptr_t req_handle) {
        auto mm = reqs_.get(req_handle);
        if (!mm) {
            return;
        }
        delete mm.value();
        reqs_.remove(req_handle);
    }

    inline void
    setDescUpdateTime(uintptr_t req_handle, uintptr_t desc_handle, uint64_t update_time) {
        auto mm = reqs_.get(req_handle);
        assert(mm);
        mm.value()->update(desc_handle,
                           [update_time](descMeta &meta) { meta.updateTime = update_time; });
    }

    inline void
    removeDescUpdateRecord(uintptr_t req_handle, uintptr_t desc_handle) {
        auto mm = reqs_.get(req_handle);
        assert(mm);
        mm.value()->remove(desc_handle);
    }

    inline uint64_t
    getDescUpdateTime(uintptr_t req_handle, uintptr_t desc_handle) {
        auto mm = reqs_.get(req_handle);
        assert(mm);
        auto desc = mm.value()->get(desc_handle);
        return desc ? desc.value().updateTime : 0;
    }

private:
    // underlying map
    gismoConcurrentMap<uintptr_t, gismoConcurrentMap<uintptr_t, descMeta> *> reqs_;
};

#endif //__GISMO_REQ_MGR_H