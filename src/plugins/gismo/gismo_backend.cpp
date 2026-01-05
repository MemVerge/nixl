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
#include <cassert>
#include <cctype>
#include <atomic>
#include <cstdint>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "gismo_backend.h"
#include "gismo_log.h"
#include "common/nixl_log.h"
#include "file/file_utils.h"
#include "gismo_utils.h"

nixlGismoEngine::nixlGismoEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params),
      requireStop_(false) {
    utils_ = new gismoUtils(this->localAgent.c_str());
    rpcHandler_ = new gismoRpcHandler(utils_);

    this->initErr = false;
    if (utils_->initMVFS(init_params) == NIXL_ERR_BACKEND) {
        NIXL_ERROR << "Error initialize mvfs";
        this->initErr = true;
        return;
    }

    rpcHandler_->start(this);

    NIXL_DEBUG << "Gismo backend initialized " << this->localAgent;
}

nixl_status_t
nixlGismoEngine::getConnInfo(std::string &str) const {
    str = "no-need-to-return";
    return NIXL_SUCCESS;
}

nixlGismoBackendMD *
nixlGismoEngine::getBackendMD(uintptr_t start_addr) const {
    auto md = memRegInfo_.get(start_addr);
    if (md.has_value()) {
        return md.value();
    }
    return nullptr;
}

nixl_status_t
nixlGismoEngine::registerMem(const nixlBlobDesc &mem,
                             const nixl_mem_t &nixl_mem,
                             nixlBackendMD *&out) {
    nixl_status_t status = NIXL_SUCCESS;

    switch (nixl_mem) {
    case DRAM_SEG:
    case VRAM_SEG: {
        auto md = memRegInfo_.get(mem.addr);
        if (md.has_value()) {
            md.value()->ref_cnt_++;
            out = md.value();
            return NIXL_SUCCESS;
        }
        bool gpu = nixl_mem == VRAM_SEG ? true : false;
        nixlGismoBackendMD *md_ptr = nullptr;
        int err = utils_->registerLocalMemory(gpu, mem.addr, mem.len, md_ptr);
        if (err) return NIXL_ERR_BACKEND;
        out = md_ptr;
        memRegInfo_.put(mem.addr, md_ptr);
        NIXL_INFO << "Gismo: Registered regular memory(addr: " << mem.addr << ", len: " << mem.len
                  << ") in " << this->localAgent << ":" << (uintptr_t)this;
        break;
    }
    case FILE_SEG: {
        int fd = mem.devId;
        std::string fp = "";
        nixlGismoBackendMD *md = new nixlGismoBackendMD(true);
        auto ret = utils_->registerFileHandle(fd, fp);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to register fd " << fd << " to Gismo";
            return NIXL_ERR_BACKEND;
        }
        NIXL_INFO << "Registered fd " << fd << " to file " << fp << " with size " << mem.len;
        md->fileHandle_ = new gismoFileHandle(fd, mem.len, mem.metaInfo, fp);
        out = (nixlBackendMD *)md;
        memRegInfo_.put((uintptr_t)md->fileHandle_, md); // record it also in map;
        break;
    }
    default:
        GISMO_LOG_RETURN(NIXL_ERR_BACKEND, "Error - type not supported");
    }

    return status;
}

nixl_status_t
nixlGismoEngine::deregisterMem(nixlBackendMD *meta) {
    auto mem_md = dynamic_cast<nixlGismoBackendMD *>(meta);
    if (mem_md != nullptr) {
        // file segment
        if (mem_md->fileHandle_ != nullptr) {
            auto exist_md = memRegInfo_.get((uintptr_t)mem_md->fileHandle_);
            if (exist_md.has_value()) {
                utils_->unregisterFileHandle(mem_md->fileHandle_->fd_,
                                             mem_md->fileHandle_->mount_point_);
                memRegInfo_.remove((uintptr_t)mem_md->fileHandle_);
                NIXL_INFO << "Unregistered fd " << mem_md->fileHandle_->fd_ << " from file "
                          << mem_md->fileHandle_->mount_point_ << " in " << this->localAgent;
            }
        } else {
            // memory segment
            auto exist_md = memRegInfo_.get(mem_md->addr_);
            if (exist_md.has_value()) {
                utils_->unregisterLocalMemory(mem_md->addr_, rpcHandler_);
                memRegInfo_.remove(mem_md->addr_);
                NIXL_INFO << "Unregistered mem " << mem_md->addr_ << " in " << this->localAgent;
            }
        }
    } else {
        return NIXL_ERR_BACKEND;
    }
    delete meta;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGismoEngine::prepXfer(const nixl_xfer_op_t &operation,
                          const nixl_meta_dlist_t &local,
                          const nixl_meta_dlist_t &remote,
                          const std::string &remote_agent,
                          nixlBackendReqH *&handle,
                          const nixl_opt_b_args_t *opt_args) const {
    // in case double prepare, do nothing
    if (utils_->getReqMgr()->containsReq((uintptr_t)handle)) {
        return NIXL_SUCCESS;
    }
    uintptr_t tmp_handle = 0; // in case of exception
    try {
        auto req_handle = std::make_unique<nixlGismoBackendReqH>(
            operation, local, remote, remote_agent, opt_args);
        tmp_handle = (uintptr_t)req_handle.get();
        utils_->getReqMgr()->addReq(tmp_handle);
        if (operation == NIXL_READ && remote_agent != this->localAgent) {
            // need to notify remote agent to refresh
            // incoming request consist of remote desc from different backends
            // so we need to group them by backend addr
            std::unordered_map<uintptr_t, std::vector<memoryMeta> *> desc_map;
            std::unordered_map<uintptr_t, bool> distinct_desc_map;
            for (auto &rds : remote) {
                NIXL_DEBUG << "Try to get backend md for " << rds.addr;
                // try to load metadata if not exist
                if (rds.metadataP == nullptr) {
                    nixlBackendMD *tmp_md = nullptr;
                    const_cast<nixlGismoEngine *>(this)->loadRemoteMD(
                        nixlBlobDesc(rds.addr, rds.len, rds.devId, ""),
                        DRAM_SEG,
                        remote_agent,
                        tmp_md);
                    if (tmp_md == nullptr) {
                        NIXL_ERROR << "Failed to load remote MD for desc addr " << rds.addr;
                        continue;
                    }
                    const_cast<nixlMetaDesc *>(&rds)->metadataP = tmp_md;
                }
                auto backend = utils_->getBackendMDFromDesc(rds);
                auto desc_list = desc_map[backend->addr_];
                if (desc_list == nullptr) {
                    desc_list = new std::vector<memoryMeta>();
                    desc_map[backend->addr_] = desc_list;
                }
                // make sure only distinct desc is added to flush request
                if (distinct_desc_map.count(rds.addr) == 0) {
                    distinct_desc_map[rds.addr] = true;
                    desc_list->push_back(memoryMeta{
                        .addr_ = rds.addr,
                        .len_ = rds.len,
                    });
                } else {
                    NIXL_WARN << "Skip duplicated desc addr " << rds.addr;
                }
            }
            for (auto &itr : desc_map) {
                auto start_addr = itr.first;
                rpcHandler_->requestMemoryFlush(req_handle.get(), start_addr, *itr.second);
                delete itr.second;
            }
        }
        handle = req_handle.release();
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Unexpected error: " << e.what();
        utils_->getReqMgr()->removeReq(tmp_handle);
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGismoEngine::postXfer(const nixl_xfer_op_t &operation,
                          const nixl_meta_dlist_t &local,
                          const nixl_meta_dlist_t &remote,
                          const std::string &remote_agent,
                          nixlBackendReqH *&handle,
                          const nixl_opt_b_args_t *opt_args) const {
    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << "Invalid arguements, not support different size post";
        return NIXL_ERR_BACKEND;
    }
    try {
        auto req_handle = dynamic_cast<nixlGismoBackendReqH *>(handle);
        assert(req_handle != nullptr);
        // if this is agent internal post, do offloading
        if (req_handle->remote_agent_ == this->localAgent) {
            auto status = utils_->offload(req_handle);
            if (status < 0) {
                NIXL_ERROR << "Error in offloading data, error " << status;
                return NIXL_ERR_BACKEND;
            }
            return (nixl_status_t)status;
        }
        auto status = utils_->transfer(req_handle, rpcHandler_);
        if (status < 0) {
            NIXL_ERROR << "Error in transfering data, error " << status;
            return NIXL_ERR_BACKEND;
        }
        NIXL_DEBUG << "Completed postXfer " << (uintptr_t)req_handle << ", total desc "
                   << req_handle->local_.descCount();
        return (nixl_status_t)status;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << e.what();
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGismoEngine::checkXfer(nixlBackendReqH *handle) const {
    auto xq = dynamic_cast<nixlGismoBackendReqH *>(handle);
    bool done = utils_->isRequestCompleted(xq, nullptr);
    NIXL_DEBUG << "CheckXfer status of " << (uintptr_t)xq << ", total " << xq->local_.descCount()
               << " " << done;
    return done ? NIXL_SUCCESS : NIXL_IN_PROG;
}

nixl_status_t
nixlGismoEngine::releaseReqH(nixlBackendReqH *handle) const {
    NIXL_DEBUG << "Ready to release handle " << (uintptr_t)handle;
    utils_->getReqMgr()->removeReq((uintptr_t)handle);
    delete handle;
    return NIXL_SUCCESS;
}

nixlGismoEngine::~nixlGismoEngine() {
    requireStop_.store(true);
    // deinit mvfs, so thread pool is gone
    if (utils_) {
        utils_->deinitMVFS();
    }
    // then clean other stuff
    if (rpcHandler_) {
        delete rpcHandler_;
    }
    if (utils_) {
        delete utils_;
    }
}

nixl_status_t
nixlGismoEngine::queryMem(const nixl_reg_dlist_t &descs,
                          std::vector<nixl_query_resp_t> &resp) const {
    // Extract metadata from descriptors which are file names
    // Different plugins might customize parsing of metaInfo to get the file names
    std::vector<nixl_blob_t> metadata(descs.descCount());
    for (int i = 0; i < descs.descCount(); ++i)
        metadata[i] = descs[i].metaInfo;

    return nixl::queryFileInfoList(metadata, resp);
}

nixl_status_t
nixlGismoEngine::getPublicData(const nixlBackendMD *meta, std::string &str) const {
    // TODO: what should we return?
    str = "empty-public-data";
    return NIXL_SUCCESS;
}

// Deserialize from string the connection info for a remote node, if supported
// The generated data should be deleted in nixlBackendEngine destructor
nixl_status_t
nixlGismoEngine::loadRemoteConnInfo(const std::string &remote_agent,
                                    const std::string &remote_conn_info) {
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGismoEngine::findMD(nixlBackendMD *&output,
                        const std::vector<nixlGismoBackendMD *> &backend_md_list,
                        uintptr_t addr) {
    for (auto &md : backend_md_list) {
        if (addr >= md->addr_ && addr < md->addr_ + md->length_) {
            output = md;
            return NIXL_SUCCESS;
        }
    };
    return NIXL_ERR_NOT_FOUND;
}

nixl_status_t
nixlGismoEngine::loadRemoteMD(const nixlBlobDesc &input,
                              const nixl_mem_t &nixl_mem,
                              const std::string &remote_agent,
                              nixlBackendMD *&output) {
    std::vector<nixlGismoBackendMD *> backend_md_list;
    auto ret = utils_->loadRemoteMemInfo(remote_agent, backend_md_list);
    if (ret < 0) {
        NIXL_ERROR << "Failed to get memory register info from " << remote_agent << " in "
                   << localAgent;
        return NIXL_ERR_BACKEND;
    }

    auto find_ret = findMD(output, backend_md_list, input.addr);
    if (find_ret != NIXL_SUCCESS) {
        NIXL_WARN << "No corresponding md found for " << input.addr << " belonging to "
                  << remote_agent << " in " << localAgent;
        backend_md_list.clear();
        // try reload once
        ret = utils_->loadRemoteMemInfo(remote_agent, backend_md_list, true);
        if (ret < 0) {
            NIXL_ERROR << "Failed to reload memory register info from " << remote_agent << " in "
                       << localAgent;
            return NIXL_ERR_BACKEND;
        }
        find_ret = findMD(output, backend_md_list, input.addr);
        if (find_ret != NIXL_SUCCESS) {
            NIXL_ERROR << "No corresponding md found for " << input.addr << " belonging to "
                       << remote_agent << " in " << localAgent;
            return NIXL_ERR_NOT_FOUND;
        }
    }
    return NIXL_SUCCESS;
}

// Populate an empty received notif list. Elements are released within backend then.
nixl_status_t
nixlGismoEngine::getNotifs(notif_list_t &notif_list) {
    // get all notif list
    rpcHandler_->pollNotifs(notif_list);
    return NIXL_SUCCESS;
}

// Generates a standalone notification, not bound to a transfer.
nixl_status_t
nixlGismoEngine::genNotif(const std::string &remote_agent, const std::string &msg) const {
    NIXL_INFO << "Gen notify for " << remote_agent << ", " << msg;
    return NIXL_SUCCESS;
}
