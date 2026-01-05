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

#ifndef __GISMO_BACKEND_H
#define __GISMO_BACKEND_H

#include <cstdint>
#include <nixl.h>
#include <nixl_types.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>

#include "backend/backend_engine.h"
#include "gismo_utils.h"
#include "gismo_concurrent_map.h"
#include "gismo_rpc_handler.h"

class nixlGismoEngine : public nixlBackendEngine {
private:
    gismoUtils *utils_ = nullptr;
    gismoRpcHandler *rpcHandler_ = nullptr;
    // record memory register info
    gismoConcurrentMap<uintptr_t, nixlGismoBackendMD *> memRegInfo_;

    std::atomic_bool requireStop_;

    nixl_status_t findMD(nixlBackendMD *&output,
        const std::vector<nixlGismoBackendMD *> &backend_md_list,
        uintptr_t addr);

public:
    nixlGismoEngine(const nixlBackendInitParams *init_params);
    ~nixlGismoEngine();

    // read/write operation needs to notify target nodes
    //
    bool
    supportsNotif() const override {
        return true;
    }

    bool
    supportsRemote() const override {
        return true; // support transfering mem to other node
    }

    bool
    supportsLocal() const override {
        return true; // support offloading mem to DMO
    }

    bool
    supportsProgTh() const {
        return false;
    }

    nixl_mem_list_t
    getSupportedMems() const override {
        return {FILE_SEG, DRAM_SEG, VRAM_SEG};
    }

    nixl_status_t
    getConnInfo(std::string &str) const override;

    nixl_status_t
    connect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    disconnect(const std::string &remote_agent) override {
        return NIXL_SUCCESS;
    }

    nixl_status_t
    loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override {
        output = input;
        return NIXL_SUCCESS;
    }

    nixl_status_t
    unloadMD(nixlBackendMD *input) override {
        // do nothing as the md will be deleted in deregisterMem
        return NIXL_SUCCESS;
    }

    nixl_status_t
    getPublicData(const nixlBackendMD *meta, std::string &str) const override;

    nixl_status_t
    loadRemoteMD(const nixlBlobDesc &input,
                 const nixl_mem_t &nixl_mem,
                 const std::string &remote_agent,
                 nixlBackendMD *&output) override;

    // Deserialize from string the connection info for a remote node, if supported
    // The generated data should be deleted in nixlBackendEngine destructor
    nixl_status_t
    loadRemoteConnInfo(const std::string &remote_agent, const std::string &remote_conn_info) override;

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;
    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;
    nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

    // Populate an empty received notif list. Elements are released within backend then.
    nixl_status_t
    getNotifs(notif_list_t &notif_list) override;

    // Generates a standalone notification, not bound to a transfer.
    virtual nixl_status_t
    genNotif(const std::string &remote_agent, const std::string &msg) const override;
    nixlGismoBackendMD *
    getBackendMD(uintptr_t start_addr) const;

    bool
    requiringStop() const {
        return requireStop_.load();
    }

    const char *
    getAgentName() const {
        return this->localAgent.c_str();
    };
};

#endif //__GISMO_BACKEND_H
