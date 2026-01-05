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
#ifndef __GISMO_UTILS_H
#define __GISMO_UTILS_H

#include <cstdint>
#include <ctime>
#include <fcntl.h>
#include <unistd.h>
#include <nixl.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <shared_mutex>

#include "backend_engine.h"

#include "mvfs.h"
#include "gismo_thread_pool.h"
#include "gismo_req_mgr.h"

#define DEFAULT_CHUNK_SIZE 2U << 20U

// define some request types
typedef uint8_t request_type;
// request to flush segments of one registered memory
// this is used before one read operation
static const request_type FLUSH_SEGMENTS = 1;
// request to load segments of one registered memory
// this is used when one write operation completed
static const request_type LOAD_SEGMENTS = 2;
// ack read segments completed
// this is used when read operation completed
static const request_type READ_SEGMENTS_ACK = 3;
// one file has been deleted
static const request_type DELETE_FILE = 4;

// define memory meta exchange between nodes
struct memoryMeta {
    uintptr_t addr_;
    size_t len_;
};

struct memoryMetaResp {
    uintptr_t addr_;
    size_t len_;
    uint8_t rslt; // read/write result
};

struct gismoFileHandle {
    gismoFileHandle(int fd, size_t sz, std::string md, std::string mp)
        : fd_(fd),
          size_(sz),
          metadata_(md),
          mount_point_(mp) {};
    int fd_;
    // -1 means inf size file?
    size_t size_;
    std::string metadata_;
    std::string mount_point_;
};

struct nixlGismoBackendMD : public nixlBackendMD {
    nixlGismoBackendMD(bool isPrivate) : nixlBackendMD(isPrivate) {}

    nixlGismoBackendMD(nixlGismoBackendMD *rhs) : nixlBackendMD(false) {
        addr_ = rhs->addr_;
        length_ = rhs->length_;
        ref_cnt_ = rhs->ref_cnt_;
        gpu_ = rhs->gpu_;
        if (rhs->fileHandle_ != nullptr) {
            fileHandle_ = new gismoFileHandle(rhs->fileHandle_->fd_,
                                              rhs->fileHandle_->size_,
                                              rhs->fileHandle_->metadata_,
                                              rhs->fileHandle_->mount_point_);
        } else {
            fileHandle_ = nullptr;
        }
    }

    virtual ~nixlGismoBackendMD() {
        if (fileHandle_ != nullptr) {
            delete fileHandle_;
            fileHandle_ = nullptr;
        }
    }

    uintptr_t addr_ = 0; // start pos of this memory
    size_t length_ = 0; // length of this block
    int ref_cnt_ = 0; // ref count of this mem
    bool gpu_ = false; // when from gpu

    // indicate whether this is a file
    gismoFileHandle *fileHandle_ = nullptr;

    // serialize this object to a string
    std::string
    toString();
    // deserialize from a string
    void
    fromString(std::string &objInStr);
};

class nixlGismoBackendReqH : public nixlBackendReqH {
public:
    const nixl_xfer_op_t &operation_; // The transfer operation (read/write)
    const nixl_meta_dlist_t &local_; // Local memory descriptor list
    const nixl_meta_dlist_t &remote_; // Remote memory descriptor list
    const std::string &remote_agent_; // remote agent id
    // as incoming opt_args may be temporary object, we need to copy it
    nixl_opt_b_args_t *opt_args_; // Optional backend-specific arguments

public:
    nixlGismoBackendReqH(const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         const std::string &remote_agent,
                         const nixl_opt_b_args_t *opt_args)
        : operation_(operation),
          local_(local),
          remote_(remote),
          remote_agent_(remote_agent) {
        if (opt_args != nullptr) {
            opt_args_ = new nixl_opt_b_args_t(*opt_args);
        } else {
            opt_args_ = new nixl_opt_b_args_t();
        }
    }

    virtual ~nixlGismoBackendReqH() {
        delete opt_args_;
        opt_args_ = nullptr;
    };

    // Exception classes
    class exception : public std::exception {
    private:
        const nixl_status_t code_;

    public:
        exception(const std::string &msg, nixl_status_t code) : std::exception(), code_(code) {}

        nixl_status_t
        code() const noexcept {
            return code_;
        }
    };
};

struct metricsItem {
    uint64_t total_count_ = 0;
    double total_time_ = 0;
    double total_write_time_ = 0;
};

struct openFdMeta {
    mvfs_handle fd_ = 0;
    uintptr_t mapped_addr_ = 0;
    size_t mapped_len_ = 0;
};

struct gismoConfigParams {
    std::string socket_path_ = "/tmp/dmo.daemon.sock.0";
    std::string client_log_path_ = "/tmp/mvfs_client.log";
    size_t thread_pool_size_ = 8;
    size_t chunk_size_ = DEFAULT_CHUNK_SIZE;
    bool record_metrics_ = false;
    bool use_mmap_ = false; // whether to use mmap for file access
};

class gismoRpcHandler;

class gismoUtils {
private:
    gismoConfigParams configParams_;
    bool hasCudaDevices_ = false; // whether this node has cuda devices
    struct mvfs_connection *connection_ = nullptr;
    struct gsmb_message_bus *mvMessageBus_ = nullptr;
    gismoThreadPool *threadPool_ = nullptr;
    gismoReqMgr *reqMgr_ = nullptr;
    const char *localAgentId_;
    std::shared_mutex memoryMtx_; // protect memory registration
    gismoConcurrentMap<std::string, openFdMeta *> openedMvfsFds_; // cache all opened fds
    gismoConcurrentMap<uint32_t, metricsItem> perfData_; // record performance data
    // record remote memory info
    gismoConcurrentMap<std::string, std::pair<uint64_t, std::vector<nixlGismoBackendMD *>>>
        remoteMemInfo_;

public:
    const char *ROOT_DIR = "/nixl/";
    const char *MEM_REGISTER_INFO = "register_info";
    const char *SOCKET_PATH = "socket_path";
    const char *SOCKET_PATH_ENV = "GISMO_SOCKET_PATH";
    const char *CLIENT_LOG_PATH = "client_log_path";
    const char *THREAD_POOL_SIZE = "thread_pool_size";
    const char *THREAD_POOL_SIZE_ENV = "GISMO_THREAD_POOL_SIZE";
    const char *RECORD_METRICS = "record_metrics";
    const char *CHUNK_SIZE = "chunk_size";
    const char *CHUNK_SIZE_ENV = "GISMO_CHUNK_SIZE";
    const char *USE_MMAP = "use_mmap";
    const char *USE_MMAP_ENV = "GISMO_USE_MMAP";

    gismoUtils(const char *id);

    ~gismoUtils();

    int
    initMVFS(const nixlBackendInitParams *p);
    void
    deinitMVFS();

    int
    registerLocalMemory(bool gpu, uintptr_t addr, size_t size, nixlGismoBackendMD *&md);
    int
    unregisterLocalMemory(uintptr_t addr, gismoRpcHandler *rpc_handler);

    int
    pollMemoryRegisterInfo(const std::string &remote_agent,
                           std::vector<nixlGismoBackendMD *> &backend_list);

    int
    loadRemoteMemInfo(const std::string &remote_agent,
                      std::vector<nixlGismoBackendMD *> &backend_md_list,
                      bool force_reload = false);

    nixl_status_t
    registerFileHandle(int fd, std::string &fp);
    int
    unregisterFileHandle(int fd, const std::string &fp);

    int
    transfer(nixlGismoBackendReqH *handle, gismoRpcHandler *);

    int
    offload(nixlGismoBackendReqH *handle);

    nixl_status_t
    offloadDesc(const nixl_xfer_op_t &operation,
                const nixlMetaDesc &src_desc,
                const nixlMetaDesc &dst_desc);

    int
    allocFile(const std::string &dst_file, size_t len);

    int
    writeFile(const std::string &dst_file, bool gpu, uintptr_t src_data, int offset, size_t len);

    size_t
    doFileWrite(openFdMeta *fd_meta, bool gpu, uintptr_t src_data, int offset, size_t len) const;

    int
    writeFileSegments(const std::string &dst_file,
                      bool gpu,
                      const uintptr_t start_addr,
                      const std::vector<memoryMeta> *segments);
    int
    readFileSegments(const std::string &dst_file,
                     bool gpu,
                     const uintptr_t start_addr,
                     const std::vector<memoryMeta> *segments);

    int
    readFile(const std::string &dst_file, bool gpu, uintptr_t src_data, size_t len, int offset);

    int
    doFileRead(openFdMeta *fd_meta, bool gpu, uintptr_t dst_buf, int offset, size_t len) const;

    int
    getFileAttribute(const char *fpath, mvfs_file_attr *attr);

    bool
    fileExists(const char *fpath);

    int
    putSmallFile(const char *fpath, const char *content, const size_t len);
    // append content to a file
    int
    appendFile(const char *fpath, const char *content, const size_t len);
    int
    getSmallFile(const char *fpath, char *content, int len);

    std::string
    generateFileForMemory(const char *agent_path, uintptr_t addr);

    int
    mkdir(const char *fp);
    // create one empty file
    int
    mknode(const char *fp);

    int
    rmnode(const char *fp);

    int
    rmdir(const char *fp);

    void
    transferDesc(nixlBackendReqH *req, const nixlMetaDesc &desc) {
        reqMgr_->transferDesc((uintptr_t)req, desc.addr);
    };

    void
    markDescDone(nixlBackendReqH *req, const nixlMetaDesc &desc) {
        reqMgr_->markDescDone((uintptr_t)req, (uintptr_t)desc.addr);
    };

    bool
    isDescCompleted(nixlBackendReqH *req, const nixlMetaDesc &desc) {
        return reqMgr_->isDescDone((uintptr_t)req, (uintptr_t)desc.addr);
    };

    gismoReqMgr *
    getReqMgr() {
        return reqMgr_;
    };

    bool
    isRequestCompleted(nixlGismoBackendReqH *req, const nixlMetaDesc *ongoing_desc);

    inline gismoThreadPool *
    getThreadPool() {
        return threadPool_;
    }

    nixlGismoBackendMD *
    getBackendMDFromDesc(const nixlMetaDesc &desc) const {
        assert(desc.metadataP != nullptr);
        auto backend = dynamic_cast<nixlGismoBackendMD *>(desc.metadataP);
        return backend;
    }

    gsmb_message_bus *
    getMessageBus() {
        return mvMessageBus_;
    }

    int
    listAllAgents(std::vector<std::string> &agents);

    void
    deleteFileCache(const char *fp);

    bool
    needRecordMetrics() const {
        return configParams_.record_metrics_;
    };

private:
    int
    initConfigParams(const nixlBackendInitParams *p);
    openFdMeta *
    getMvfsHandle(const char *fp, bool preload = false);
    void
    removeLineFromFile(const char *fp, const char *line_starter);
    void
    writeRemoteDesc(const nixlMetaDesc &dms,
                    const nixlMetaDesc &sms,
                    nixlGismoBackendReqH *handle,
                    gismoRpcHandler *rpc_handler);
    void
    readRemoteDesc(const nixlMetaDesc &dms,
                   const nixlMetaDesc &sms,
                   nixlGismoBackendReqH *handle,
                   gismoRpcHandler *rpc_handler);
};

uintptr_t
strToInt(const std::string &str);

long unsigned int
getCurrentTime();

#endif //__GISMO_UTILS_H
