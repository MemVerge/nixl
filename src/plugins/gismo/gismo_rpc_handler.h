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

#ifndef __GISMO_RPC_HANDLER_H
#define __GISMO_RPC_HANDLER_H

#include <cstddef>
#include <cstdint>
#include <cassert>
#include <atomic>
#include <cstring>
#include <sys/types.h>

#include "backend_engine.h"

#include "gismo_concurrent_map.h"
#include "gismo_utils.h"
#include "gismo_blocking_list.h"
#include "gsmb/gsmb_c.h"

class nixlGismoEngine;

class rpcRequest {
public:
    uint64_t req_id_;
    std::string src_agent_;
    rpcRequest(uint64_t ri, const char *sa) : req_id_(ri), src_agent_(sa) {};
    virtual ~rpcRequest() = default;
};

class deleteFileRequest : public rpcRequest {
public:
    std::string filePath;
    deleteFileRequest(uint64_t ri, const char *sa, const char *fp)
        : rpcRequest(ri, sa),
          filePath(fp) {};
    ~deleteFileRequest() {};
};

class flushSegmentRequest : public rpcRequest {
public:
    uintptr_t req_handle_; // record the request handle from initiator
    uintptr_t start_addr_;
    std::vector<memoryMeta> *mem_list_;
    flushSegmentRequest(uint64_t ri,
                        const char *sa,
                        uintptr_t rh,
                        uintptr_t addr,
                        std::vector<memoryMeta> *ml)
        : rpcRequest(ri, sa),
          req_handle_(rh),
          start_addr_(addr),
          mem_list_(ml) {};

    ~flushSegmentRequest() {
        delete mem_list_;
    };
};

class loadSegmentRequest : public rpcRequest {
public:
    uintptr_t start_addr_;
    uintptr_t req_handle_; // record the request handle from initiator
    size_t total_desc_len_; // record the total desc length in request
    std::string notify_msg_; // record the notify msg from initiator
    std::vector<memoryMeta> *mem_list_;
    loadSegmentRequest(uint64_t ri,
                       const char *sa,
                       uintptr_t rh,
                       uintptr_t addr,
                       size_t tlen,
                       const char *msg,
                       std::vector<memoryMeta> *ml)
        : rpcRequest(ri, sa),
          start_addr_(addr),
          req_handle_(rh),
          total_desc_len_(tlen),
          notify_msg_(msg),
          mem_list_(ml) {};

    ~loadSegmentRequest() {
        delete mem_list_;
    }
};

class readSegmentAckRequest : public rpcRequest {
public:
    uintptr_t start_addr_;
    uintptr_t req_handle_; // record the request handle from initiator
    size_t total_desc_len_; // record the total desc length in request
    std::string notify_msg_; // record the notify msg from initiator
    std::vector<memoryMetaResp> *mem_list_;
    readSegmentAckRequest(uint64_t ri,
                          const char *sa,
                          uintptr_t addr,
                          uintptr_t rh,
                          size_t tlen,
                          const std::string &msg,
                          std::vector<memoryMetaResp> *ml)
        : rpcRequest(ri, sa),
          start_addr_(addr),
          req_handle_(rh),
          total_desc_len_(tlen),
          notify_msg_(msg),
          mem_list_(ml) {};

    ~readSegmentAckRequest() {
        delete mem_list_;
    }
};

class rpcResponse {
public:
    uint64_t rsp_id_;
    std::string src_agent_;

    rpcResponse(uint64_t ri, const char *sa) : rsp_id_(ri), src_agent_(sa) {}

    virtual ~rpcResponse() = default;
};

class flushSegmentResponse : public rpcResponse {
public:
    uintptr_t req_handle_;
    uintptr_t start_addr_;
    std::vector<memoryMetaResp> *mem_list_;
    flushSegmentResponse(uint64_t ri,
                         const char *sa,
                         uintptr_t rh,
                         uintptr_t addr,
                         std::vector<memoryMetaResp> *ml)
        : rpcResponse(ri, sa),
          req_handle_(rh),
          start_addr_(addr),
          mem_list_(ml) {};

    ~flushSegmentResponse() {
        delete mem_list_;
    }
};

class loadSegmentResponse : public rpcResponse {
public:
    uintptr_t start_addr_;
    std::vector<memoryMetaResp> *mem_list_;
    loadSegmentResponse(uint64_t ri,
                        const char *sa,
                        uintptr_t addr,
                        std::vector<memoryMetaResp> *ml)
        : rpcResponse(ri, sa),
          start_addr_(addr),
          mem_list_(ml) {};

    ~loadSegmentResponse() {
        delete mem_list_;
    }
};

class rpcMetrics {
private:
    std::atomic_uint64_t recvdReqCount_ = 0;
    std::atomic_uint64_t recvdRespCount_ = 0;
    std::atomic_uint64_t sentRespCount_ = 0;
    std::atomic_uint64_t sentReqCount_ = 0;

    std::atomic_uint64_t sentFlushSegReqCount_ = 0;
    std::atomic_uint64_t sentLoadSegReqCount_ = 0;
    std::atomic_uint64_t sentReadSegAckReqCount_ = 0;
    std::atomic_uint64_t sentDeleteFileReqCount_ = 0;
    std::atomic_uint64_t recvdFlushSegReqCount_ = 0;
    std::atomic_uint64_t recvdLoadSegReqCount_ = 0;
    std::atomic_uint64_t recvdReadSegAckReqCount_ = 0;
    std::atomic_uint64_t recvdDeleteFileReqCount_ = 0;
    std::atomic_uint64_t sentFlushSegRespCount_ = 0;
    std::atomic_uint64_t sentLoadSegRespCount_ = 0;
    std::atomic_uint64_t recvdFlushSegRespCount_ = 0;
    std::atomic_uint64_t recvdLoadSegRespCount_ = 0;

public:
    void
    increaseSentReqCount(request_type rt) {
        sentReqCount_.fetch_add(1);
        switch (rt) {
        case FLUSH_SEGMENTS:
            sentFlushSegReqCount_.fetch_add(1);
            break;
        case LOAD_SEGMENTS:
            sentLoadSegReqCount_.fetch_add(1);
            break;
        case READ_SEGMENTS_ACK:
            sentReadSegAckReqCount_.fetch_add(1);
            break;
        case DELETE_FILE:
            sentDeleteFileReqCount_.fetch_add(1);
            break;
        default:
            break;
        }
    }

    void
    increaseRecvdReqCount(request_type rt) {
        recvdReqCount_.fetch_add(1);
        switch (rt) {
        case FLUSH_SEGMENTS:
            recvdFlushSegReqCount_.fetch_add(1);
            break;
        case LOAD_SEGMENTS:
            recvdLoadSegReqCount_.fetch_add(1);
            break;
        case READ_SEGMENTS_ACK:
            recvdReadSegAckReqCount_.fetch_add(1);
            break;
        case DELETE_FILE:
            recvdDeleteFileReqCount_.fetch_add(1);
            break;
        default:
            break;
        }
    }

    void
    increaseSentRespCount(request_type rt) {
        sentRespCount_.fetch_add(1);
        switch (rt) {
        case FLUSH_SEGMENTS:
            sentFlushSegRespCount_.fetch_add(1);
            break;
        case LOAD_SEGMENTS:
            sentLoadSegRespCount_.fetch_add(1);
            break;
        default:
            break;
        }
    }

    void
    increaseRecvdRespCount(request_type rt) {
        recvdRespCount_.fetch_add(1);
        switch (rt) {
        case FLUSH_SEGMENTS:
            recvdFlushSegRespCount_.fetch_add(1);
            break;
        case LOAD_SEGMENTS:
            recvdLoadSegRespCount_.fetch_add(1);
            break;
        default:
            break;
        }
    }
};

class gismoRpcHandler {
public:
    inline gismoRpcHandler(gismoUtils *u) : utils_(u) {};

    ~gismoRpcHandler();

    void
    start(const nixlGismoEngine *engine);

    // request one remote agent to flush specified memory segments
    void
    requestMemoryFlush(const nixlGismoBackendReqH *req_handle,
                       const uintptr_t start_addr,
                       const std::vector<memoryMeta> &mem_list);
    // request one remote agent to load specified memory segments;
    void
    requestMemoryLoad(const nixlGismoBackendReqH *req_handle,
                      const uintptr_t start_addr,
                      const std::vector<memoryMeta> &mem_list);

    void
    requestReadSegmentAck(const nixlGismoBackendReqH *req_handle,
                          const uintptr_t start_addr,
                          const std::vector<memoryMetaResp> &mem_list);

    void
    pollNotifs(notif_list_t &notif_list);

    void
    appendNotifs(notif_list_t &new_notif_list);

    int
    updateRemoteAgent(const nixlGismoBackendReqH *req_handle,
                      const nixlMetaDesc &dms,
                      const nixlMetaDesc &sms,
                      const uintptr_t start_addr);
    int
    broadcastRequest(const gsmb_payload &pl, request_type rt);

    int
    broadcastFileDelete(const char *fp);

private:
    void
    handleRoutine();
    int
    initialize();
    // return whether handled rpc
    int
    handleRpc();
    void
    handleRequest();
    void
    handleResponses();
    void
    packFlushSegmentsReq(gsmb_payload &gp /*out*/,
                         const nixlGismoBackendReqH *req_handle /*in*/,
                         const uintptr_t start_addr /*in*/,
                         const std::vector<memoryMeta> &mem_list /*in*/);
    flushSegmentRequest *
    unpackFlushSegmentsReq(const gsmb_payload &gp);
    void
    packLoadSegmentsReq(gsmb_payload &gp /*out*/,
                        const nixlGismoBackendReqH *req_handle /*in*/,
                        const uintptr_t start_addr /*in*/,
                        const std::vector<memoryMeta> &mem_list /*in*/);
    loadSegmentRequest *
    unpackLoadSegmentsReq(const gsmb_payload &gp);
    void
    packFlushSegmentsResp(gsmb_payload &gp /*out*/,
                          const uintptr_t req_handle /*in*/,
                          const uintptr_t start_addr /*in*/,
                          const std::vector<memoryMetaResp> &mem_list /*in*/);
    flushSegmentResponse *
    unpackFlushSegmentsResp(const gsmb_payload &gp);
    void
    packLoadSegmentsResp(gsmb_payload &gp /*out*/,
                         const uintptr_t start_addr /*in*/,
                         const std::vector<memoryMetaResp> &mem_list /*in*/);
    loadSegmentResponse *
    unpackLoadSegmentsResp(const gsmb_payload &gp);

    void
    packReadSegmentsAckReq(gsmb_payload &gp /*out*/,
                           const nixlGismoBackendReqH *req_handle /*in*/,
                           const uintptr_t start_addr /*in*/,
                           const std::vector<memoryMetaResp> &mem_list /*in*/);
    readSegmentAckRequest *
    unpackReadSegmentsAckReq(const gsmb_payload &gp);

    deleteFileRequest *
    unpackDeleteFileReq(const gsmb_payload &gp);
    void
    packDeleteFileReq(gsmb_payload &gp /*out*/, const std::string &fp /*int*/);

    void
    handleLoadSegment(const loadSegmentRequest *req);
    void
    handleFlushSegment(const flushSegmentRequest *req);
    void
    handleReadSegmentAck(const readSegmentAckRequest *req);
    void
    handleDeleteFile(const deleteFileRequest *req);
    void
    handleLoadSegmentResp(const loadSegmentResponse *rsp);
    void
    handleFlushSegmentResp(const flushSegmentResponse *rsp);
    void
    unpackRequest(const uint64_t req_id, const char *src_agent, const gsmb_payload &req_payload);
    void
    unpackResponse(const uint64_t req_id, const gsmb_payload &rsp_payload, const char *src_agent);

    uint64_t
    sendWithRetry(const char *dst_agent, request_type rt, const gsmb_payload &gb);

    void
    sendCompleteResponse(request_type req_type, const uint64_t req_id);

private:
    const nixlGismoEngine *engine_;
    gsmb_message_bus *mvMessageBus_;
    gismoUtils *utils_;
    // record recvd notifications
    gismoBlockingList<std::pair<std::string, std::string>> recvdNotifList_;

    gismoBlockingList<rpcRequest *> recvdReqs_;
    gismoBlockingList<rpcResponse *> recvdResps_;

    gismoConcurrentMap<uint32_t, metricsItem> perfData_; // record performance data
    gismoConcurrentMap<uintptr_t, size_t> ongoingXferReqs_; // record ongoing requests with corresponding desc count

    rpcMetrics metrics_;

    std::atomic_int8_t reqHandlers_ = 0;
    std::atomic_int8_t respHandlers_ = 0;

    const static char EOL_FLAG = 127;
    const uint64_t SLEEP_INTERVAL = 50;
    const int RETRY_LIMIT = 10000; // retry more times in case system is busy
    const size_t MAX_PAYLOAD_SIZE = 4096 - 64;
};


#endif //__GISMO_RPC_HANDLER_H