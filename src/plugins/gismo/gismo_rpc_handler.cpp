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

#include <absl/strings/str_format.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>

#include "gismo_rpc_handler.h"
#include "common/nixl_log.h"
#include "gismo_backend.h"
#include "gismo_utils.h"

gismoRpcHandler::~gismoRpcHandler() {
    if (utils_->needRecordMetrics()) {
        NIXL_WARN << "Record for reading chunks";
        std::vector<uint32_t> pkeys;
        perfData_.keys(pkeys);
        for (auto &k : pkeys) {
            auto item = perfData_.get(k);
            assert(item);
            NIXL_WARN << "chunk_size: " << k << ", total_count: " << item.value().total_count_
                      << ", total_time: " << item.value().total_time_
                      << ", avg_time: " << item.value().total_time_ / item.value().total_count_
                      << "us";
            perfData_.remove(k);
        }
    }
};

void
gismoRpcHandler::handleRoutine() {
    do {
        auto ret = handleRpc();
        // when there is no rpc, can have a rest
        if (ret == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(SLEEP_INTERVAL));
        }
    } while (!engine_->requiringStop());
}

void
gismoRpcHandler::start(const nixlGismoEngine *engine) {
    auto ret = initialize();
    if (ret != 0) {
        NIXL_ERROR << "Failed to initialize rpc module";
        return;
    }
    engine_ = engine;
    // start one thread to handle rpc timely
    std::thread t(&gismoRpcHandler::handleRoutine, this);
    auto name = std::string(engine_->getAgentName()) + "-rpc";
    utils_->getThreadPool()->addDaemonWorker(t, name.c_str());
    NIXL_INFO << "RPC module started " << engine_->getAgentName() << ":" << (uintptr_t)engine;
}

int
gismoRpcHandler::initialize() {
    // initialize rpc module
    mvMessageBus_ = utils_->getMessageBus();
    return 0;
}

int
gismoRpcHandler::handleRpc() {
    auto messages = gsmb_receive(mvMessageBus_, GS_RECEIVE_ALL);
    for (size_t i = 0; i < messages.count && !utils_->getThreadPool()->stopping(); ++i) {
        auto m = messages.messages[i];
        if (m.is_request) {
            // new requests
            if (m.request.data == nullptr) {
                NIXL_WARN << "Received null request data from " << m.messages_from << " in "
                          << engine_->getAgentName();
                continue;
            }
            if (m.request.data[m.request.size - 1] != EOL_FLAG) {
                NIXL_WARN << "Received invalid request data from " << m.messages_from << " in "
                          << engine_->getAgentName();
                continue;
            }
            NIXL_DEBUG << "Received one request " << m.request_id << "(" << int(m.request.data[0])
                       << ") from " << m.messages_from << " owner " << m.request_owner << " in "
                       << engine_->getAgentName();
            unpackRequest(m.request_id, m.messages_from, m.request);
            // add one task to handle request when there are not many handlers
            if (reqHandlers_.load() < utils_->getThreadPool()->getStaticWorkerCount()) {
                utils_->getThreadPool()->enqueue([this]() { handleRequest(); });
            }
        } else {
            // got response
            NIXL_DEBUG << "Received response of request " << m.request_id << "(" << m.messages_from
                       << ") with owner " << m.request_owner << " in " << engine_->getAgentName();
            for (size_t i = 0; i < m.response_count; ++i) {
                auto resp = m.responses[i];
                if (resp.data == nullptr) {
                    NIXL_WARN << "Received null response data from " << m.messages_from << " in "
                              << engine_->getAgentName();
                    continue;
                }
                if (resp.data[resp.size - 1] != EOL_FLAG) {
                    NIXL_WARN << "Received invalid response data from " << m.messages_from << " in "
                              << engine_->getAgentName();
                    continue;
                }
                unpackResponse(m.request_id, resp, m.messages_from);
            }
            // add one task to handle response when there are not many handlers
            if (respHandlers_.load() < utils_->getThreadPool()->getStaticWorkerCount()) {
                utils_->getThreadPool()->enqueue([this]() { handleResponses(); });
            }
        };
    };
    auto ret = messages.count;
    gsmb_free_messages(&messages);
    return ret;
}

void
gismoRpcHandler::handleRequest() {
    reqHandlers_.fetch_add(1);
    for (auto req = recvdReqs_.pop_front(); req.has_value(); req = recvdReqs_.pop_front()) {
        NIXL_DEBUG << "Ready to handle " << req.value()->req_id_;
        auto flush_req = dynamic_cast<flushSegmentRequest *>(req.value());
        if (flush_req) {
            NIXL_DEBUG << "Ready to flush segment " << req.value()->req_id_ << ", "
                       << flush_req->start_addr_;
            handleFlushSegment(flush_req);
            delete flush_req;
            continue;
        }
        auto load_req = dynamic_cast<loadSegmentRequest *>(req.value());
        if (load_req) {
            NIXL_DEBUG << "Ready to load segment " << req.value()->req_id_ << ", "
                       << load_req->start_addr_ << " at " << engine_->getAgentName();
            handleLoadSegment(load_req);
            delete load_req;
            continue;
        }

        auto read_ack_req = dynamic_cast<readSegmentAckRequest *>(req.value());
        if (read_ack_req) {
            handleReadSegmentAck(read_ack_req);
            delete read_ack_req;
            continue;
        }
        auto delete_file_req = dynamic_cast<deleteFileRequest *>(req.value());
        if (delete_file_req) {
            handleDeleteFile(delete_file_req);
            delete delete_file_req;
            continue;
        }
        NIXL_WARN << "invalid request " << req.value()->req_id_;
        delete req.value();
    }
    reqHandlers_.fetch_sub(1);
}

void
gismoRpcHandler::unpackRequest(const uint64_t req_id,
                               const char *src_agent,
                               const gsmb_payload &req_payload) {
    rpcRequest *req = nullptr;
    switch (req_payload.data[0]) {
    case FLUSH_SEGMENTS:
        req = unpackFlushSegmentsReq(req_payload);
        break;
    case LOAD_SEGMENTS:
        req = unpackLoadSegmentsReq(req_payload);
        break;
    case READ_SEGMENTS_ACK:
        req = unpackReadSegmentsAckReq(req_payload);
        break;
    case DELETE_FILE:
        req = unpackDeleteFileReq(req_payload);
        break;
    default:
        NIXL_WARN << "Unknown request (" << int(req_payload.data[0]);
    }
    if (req) {
        metrics_.increaseRecvdReqCount(req_payload.data[0]);
        req->req_id_ = req_id;
        req->src_agent_ = src_agent;
        recvdReqs_.push_back(req);
    }
}

void
gismoRpcHandler::unpackResponse(const uint64_t req_id,
                                const gsmb_payload &rsp_payload,
                                const char *src_agent) {
    rpcResponse *rsp = nullptr;
    metrics_.increaseRecvdRespCount(rsp_payload.data[0]);
    switch (rsp_payload.data[0]) {
    case FLUSH_SEGMENTS:
        rsp = unpackFlushSegmentsResp(rsp_payload);
        break;
    case LOAD_SEGMENTS:
        rsp = unpackLoadSegmentsResp(rsp_payload);
        break;
    case READ_SEGMENTS_ACK:
        // no need to handle this ack;
        break;
    case DELETE_FILE:
        // no need to handle delete file resp;
        break;
    default:
        NIXL_WARN << "Unknown req type (" << int(rsp_payload.data[0]) << ")";
    }
    if (rsp) {
        rsp->rsp_id_ = req_id;
        rsp->src_agent_ = src_agent;
        recvdResps_.push_back(rsp);
    }
}

void
gismoRpcHandler::handleResponses() {
    respHandlers_.fetch_add(1);
    for (auto rsp = recvdResps_.pop_front(); rsp.has_value(); rsp = recvdResps_.pop_front()) {
        auto flush_rsp = dynamic_cast<flushSegmentResponse *>(rsp.value());
        if (flush_rsp) {
            NIXL_DEBUG << "Ready to handle flush segment response " << rsp.value()->rsp_id_
                       << " from " << rsp.value()->src_agent_ << " in " << engine_->getAgentName();
            handleFlushSegmentResp(flush_rsp);
            delete flush_rsp;
            continue;
        }
        auto load_rsp = dynamic_cast<loadSegmentResponse *>(rsp.value());
        if (load_rsp) {
            handleLoadSegmentResp(load_rsp);
            delete load_rsp;
            continue;
        }
        NIXL_WARN << "invalid response " << rsp.value()->rsp_id_;
        delete rsp.value();
    }
    respHandlers_.fetch_sub(1);
}

flushSegmentRequest *
gismoRpcHandler::unpackFlushSegmentsReq(const gsmb_payload &gp) {
    // The structure is
    // REQ_FLAG | REQ_HANDLE | START_ADDR | LEN_OF_MEMORY_META | MEMORY_META_LIST
    uintptr_t req_handle = 0;
    uintptr_t start_addr = 0;
    size_t len = 0;
    std::vector<memoryMeta> *mem_list = new std::vector<memoryMeta>();
    size_t pos = 1;
    assert(gp.data[gp.size - 1] == EOL_FLAG);
    memcpy(&req_handle, gp.data + pos, sizeof(uintptr_t));
    pos += sizeof(uintptr_t);
    memcpy(&start_addr, gp.data + pos, sizeof(uintptr_t));
    pos += sizeof(uintptr_t);
    memcpy(&len, gp.data + pos, sizeof(size_t));
    memoryMeta meta;
    pos += sizeof(size_t);
    for (size_t i = 0; i < len; ++i) {
        assert(pos < gp.size);
        memcpy(&meta, gp.data + pos, sizeof(memoryMeta));
        mem_list->push_back(meta);
        pos += sizeof(memoryMeta);
    }
    assert(size_t(pos) == gp.size - 1);
    return new flushSegmentRequest(0, "", req_handle, start_addr, mem_list);
}

void
gismoRpcHandler::packFlushSegmentsReq(gsmb_payload &gp,
                                      const nixlGismoBackendReqH *req_handle /*in*/,
                                      const uintptr_t start_addr,
                                      const std::vector<memoryMeta> &mem_list) {
    // The structure is
    // REQ_FLAG | REQ_HANDLE | START_ADDR | LEN_OF_MEMORY_META | MEMORY_META_LIST | EOL
    gp.size = 1 + sizeof(uintptr_t) + sizeof(uintptr_t) + sizeof(size_t) +
        sizeof(memoryMeta) * mem_list.size() + 1;
    if (gp.size >= MAX_PAYLOAD_SIZE) {
        NIXL_ERROR << "Payload size " << gp.size << " exceeds max limit " << MAX_PAYLOAD_SIZE
                   << " because it contains " << mem_list.size() << " segments in "
                   << engine_->getAgentName();
    }
    uintptr_t req_handle_ptr = (uintptr_t)req_handle;
    gp.type = GS_PAYLOAD_TYPE_REQUEST;
    gp.data = new uint8_t[gp.size];
    gp.data[0] = FLUSH_SEGMENTS;
    gp.data[gp.size - 1] = EOL_FLAG;
    int pos = 1;
    memcpy(gp.data + pos, &req_handle_ptr, sizeof(uintptr_t));
    pos += sizeof(uintptr_t);
    memcpy(gp.data + pos, &start_addr, sizeof(uintptr_t));
    pos += sizeof(uintptr_t);
    size_t len = mem_list.size();
    memcpy(gp.data + pos, &len, sizeof(size_t));
    pos += sizeof(size_t);
    for (size_t i = 0; i < mem_list.size(); ++i) {
        memcpy(gp.data + pos, &mem_list[i], sizeof(memoryMeta));
        pos += sizeof(memoryMeta);
    }
    assert(size_t(pos) == gp.size - 1);
    assert(gp.size < MAX_PAYLOAD_SIZE);
}

void
gismoRpcHandler::packLoadSegmentsReq(gsmb_payload &gp,
                                     const nixlGismoBackendReqH *req_handle,
                                     const uintptr_t start_addr,
                                     const std::vector<memoryMeta> &mem_list) {
    // The structure is
    // REQ_FLAG | REQ_HANDLE_ID | TOTAL_DESC_COUNT | MSG_LEN | MSG | START_ADDR | LEN_OF_MEMORY_META
    // | MEMORY_META_LIST |EOL
    gp.size = 1 + sizeof(uintptr_t) + sizeof(size_t) + sizeof(size_t) +
        req_handle->opt_args_->notifMsg.size() + sizeof(uintptr_t) + sizeof(size_t) +
        sizeof(memoryMeta) * mem_list.size() + 1;
    if (gp.size >= MAX_PAYLOAD_SIZE) {
        NIXL_ERROR << "Payload size " << gp.size << " exceeds max limit " << MAX_PAYLOAD_SIZE
                   << " because it contains " << mem_list.size() << " segments in "
                   << engine_->getAgentName();
    }
    gp.type = GS_PAYLOAD_TYPE_REQUEST;
    gp.data = new uint8_t[gp.size];
    gp.data[0] = LOAD_SEGMENTS;
    gp.data[gp.size - 1] = EOL_FLAG;
    int pos = 1;
    uintptr_t req_handle_ptr = (uintptr_t)req_handle;
    memcpy(gp.data + pos, &req_handle_ptr, sizeof(uintptr_t));
    pos += sizeof(uintptr_t);
    size_t total_desc_count = req_handle->local_.descCount();
    memcpy(gp.data + pos, &total_desc_count, sizeof(size_t));
    pos += sizeof(size_t);
    size_t msg_len = req_handle->opt_args_->notifMsg.size();
    memcpy(gp.data + pos, &msg_len, sizeof(size_t));
    pos += sizeof(size_t);
    memcpy(gp.data + pos,
           req_handle->opt_args_->notifMsg.c_str(),
           req_handle->opt_args_->notifMsg.size());
    pos += req_handle->opt_args_->notifMsg.size();

    memcpy(gp.data + pos, &start_addr, sizeof(uintptr_t));
    pos += sizeof(uintptr_t);
    size_t len = mem_list.size();
    memcpy(gp.data + pos, &len, sizeof(size_t));
    pos += sizeof(size_t);
    for (size_t i = 0; i < mem_list.size(); ++i) {
        memcpy(gp.data + pos, &mem_list[i], sizeof(memoryMeta));
        pos += sizeof(memoryMeta);
    }
    assert(size_t(pos) == gp.size - 1);
    assert(gp.size < MAX_PAYLOAD_SIZE);
}

loadSegmentRequest *
gismoRpcHandler::unpackLoadSegmentsReq(const gsmb_payload &gp) {
    uintptr_t start_addr = 0;
    uintptr_t req_handle_ptr = 0;
    size_t total_desc_count = 0;
    size_t msg_len = 0;
    std::string notify_msg;
    std::vector<memoryMeta> *mem_list = new std::vector<memoryMeta>();
    // The structure is
    // REQ_FLAG | REQ_HANDLE_ID | TOTAL_DESC_COUNT | MSG_LEN | MSG | START_ADDR | LEN_OF_MEMORY_META
    // | MEMORY_META_LIST
    int pos = 1;
    memcpy(&req_handle_ptr, gp.data + pos, sizeof(uintptr_t));
    pos += sizeof(uintptr_t);
    memcpy(&total_desc_count, gp.data + pos, sizeof(size_t));
    pos += sizeof(size_t);
    memcpy(&msg_len, gp.data + pos, sizeof(size_t));
    pos += sizeof(size_t);
    notify_msg.resize(msg_len);
    memcpy(&notify_msg[0], gp.data + pos, msg_len);
    pos += msg_len;

    memcpy(&start_addr, gp.data + pos, sizeof(uintptr_t));
    size_t len = 0;
    pos += sizeof(uintptr_t);
    memcpy(&len, gp.data + pos, sizeof(size_t));
    memoryMeta meta;
    pos += sizeof(size_t);
    for (size_t i = 0; i < len; ++i) {
        memcpy(&meta, gp.data + pos, sizeof(memoryMeta));
        mem_list->push_back(meta);
        pos += sizeof(memoryMeta);
    }
    assert(size_t(pos) == gp.size - 1);
    return new loadSegmentRequest(
        0, "", req_handle_ptr, start_addr, total_desc_count, notify_msg.c_str(), mem_list);
}

void
gismoRpcHandler::packFlushSegmentsResp(gsmb_payload &gp,
                                       const uintptr_t req_handle,
                                       const uintptr_t start_addr,
                                       const std::vector<memoryMetaResp> &mem_list) {
    // The structure is
    // REQ_FLAG | REQ_HANDLE | START_ADDR | LEN_OF_MEMORY_META | MEMORY_META_LIST | EOL
    gp.size = 1 + sizeof(uintptr_t) + sizeof(uintptr_t) + sizeof(size_t) +
        sizeof(memoryMetaResp) * mem_list.size() + 1;
    gp.type = GS_PAYLOAD_TYPE_REQUEST;
    gp.data = new uint8_t[gp.size];
    gp.data[0] = FLUSH_SEGMENTS;
    gp.data[gp.size - 1] = EOL_FLAG;
    int pos = 1;
    memcpy(gp.data + pos, &req_handle, sizeof(uintptr_t));
    pos += sizeof(uintptr_t);
    memcpy(gp.data + pos, &start_addr, sizeof(uintptr_t));
    pos += sizeof(uintptr_t);
    size_t len = mem_list.size();
    memcpy(gp.data + pos, &len, sizeof(size_t));
    pos += sizeof(size_t);
    for (size_t i = 0; i < mem_list.size(); ++i) {
        memcpy(gp.data + pos, &mem_list[i], sizeof(memoryMetaResp));
        pos += sizeof(memoryMetaResp);
    }
    assert(size_t(pos) == gp.size - 1);
    assert(gp.size < MAX_PAYLOAD_SIZE);
};

flushSegmentResponse *
gismoRpcHandler::unpackFlushSegmentsResp(const gsmb_payload &gp) {
    // The structure is
    // REQ_FLAG | REQ_HANDLE | START_ADDR | LEN_OF_MEMORY_META | MEMORY_META_LIST
    if (gp.size == 5) {
        // empty response
        return nullptr;
    }
    int pos = 1;
    uintptr_t req_handle = 0;
    uintptr_t start_addr = 0;
    size_t len = 0;
    std::vector<memoryMetaResp> *mem_list = new std::vector<memoryMetaResp>();

    memcpy(&req_handle, gp.data + pos, sizeof(uintptr_t));
    pos += sizeof(uintptr_t);
    memcpy(&start_addr, gp.data + pos, sizeof(uintptr_t));
    pos += sizeof(uintptr_t);
    memcpy(&len, gp.data + pos, sizeof(size_t));
    pos += sizeof(size_t);
    memoryMetaResp meta;
    for (size_t i = 0; i < len; ++i) {
        memcpy(&meta, gp.data + pos, sizeof(memoryMetaResp));
        mem_list->push_back(meta);
        pos += sizeof(memoryMetaResp);
    }
    assert(size_t(pos) == gp.size - 1);
    return new flushSegmentResponse(0, "", req_handle, start_addr, mem_list);
};

void
gismoRpcHandler::packLoadSegmentsResp(gsmb_payload &gp,
                                      const uintptr_t start_addr,
                                      const std::vector<memoryMetaResp> &mem_list) {
    // The structure is
    // REQ_FLAG | START_ADDR | LEN_OF_MEMORY_META | MEMORY_META_LIST | EOL
    gp.size = 1 + sizeof(uintptr_t) + sizeof(size_t) + sizeof(memoryMetaResp) * mem_list.size() + 1;
    gp.type = GS_PAYLOAD_TYPE_COMPLETED_RESPONSE;
    gp.data = new uint8_t[gp.size];
    gp.data[0] = LOAD_SEGMENTS;
    gp.data[gp.size - 1] = EOL_FLAG;
    int pos = 1;
    memcpy(gp.data + pos, &start_addr, sizeof(uintptr_t));
    pos += sizeof(uintptr_t);
    size_t len = mem_list.size();
    memcpy(gp.data + pos, &len, sizeof(size_t));
    pos += sizeof(size_t);
    for (size_t i = 0; i < mem_list.size(); ++i) {
        memcpy(gp.data + pos, &mem_list[i], sizeof(memoryMetaResp));
        pos += sizeof(memoryMetaResp);
    }
    assert(size_t(pos) == gp.size - 1);
    assert(gp.size < MAX_PAYLOAD_SIZE);
};

loadSegmentResponse *
gismoRpcHandler::unpackLoadSegmentsResp(const gsmb_payload &gp) {
    // The structure is
    // REQ_FLAG | START_ADDR | LEN_OF_MEMORY_META | MEMORY_META_LIST
    if (gp.size == 5) {
        // empty response
        return nullptr;
    }
    int pos = 1;
    uintptr_t start_addr = 0;
    std::vector<memoryMetaResp> *mem_list = new std::vector<memoryMetaResp>();
    memcpy(&start_addr, gp.data + pos, sizeof(uintptr_t));
    size_t len = 0;
    pos += sizeof(uintptr_t);
    memcpy(&len, gp.data + pos, sizeof(size_t));
    memoryMetaResp meta;
    pos += sizeof(size_t);
    for (size_t i = 0; i < len; ++i) {
        memcpy(&meta, gp.data + pos, sizeof(memoryMetaResp));
        mem_list->push_back(meta);
        pos += sizeof(memoryMetaResp);
    }
    assert(size_t(pos) == gp.size - 1);
    return new loadSegmentResponse(0, "", start_addr, mem_list);
};

void
gismoRpcHandler::packReadSegmentsAckReq(gsmb_payload &gp,
                                        const nixlGismoBackendReqH *req_handle,
                                        const uintptr_t start_addr,
                                        const std::vector<memoryMetaResp> &mem_list) {
    // The structure is
    // REQ_FLAG | REQ_HANDLE_ID | TOTAL_DESC_COUNT | MSG_LEN | MSG | START_ADDR | LEN_OF_MEMORY_META
    // | MEMORY_META_LIST |EOL
    gp.size = 1 + sizeof(uintptr_t) + sizeof(size_t) + sizeof(size_t) +
        req_handle->opt_args_->notifMsg.size() + sizeof(uintptr_t) + sizeof(size_t) +
        sizeof(memoryMetaResp) * mem_list.size() + 1;
    gp.type = GS_PAYLOAD_TYPE_COMPLETED_RESPONSE;
    gp.data = new uint8_t[gp.size];
    gp.data[0] = READ_SEGMENTS_ACK;
    gp.data[gp.size - 1] = EOL_FLAG;
    int pos = 1;
    uintptr_t req_handle_ptr = (uintptr_t)req_handle;
    memcpy(gp.data + pos, &req_handle_ptr, sizeof(uintptr_t));
    pos += sizeof(uintptr_t);
    size_t total_desc_count = req_handle->local_.descCount();
    memcpy(gp.data + pos, &total_desc_count, sizeof(size_t));
    pos += sizeof(size_t);
    size_t msg_len = req_handle->opt_args_->notifMsg.size();
    memcpy(gp.data + pos, &msg_len, sizeof(size_t));
    pos += sizeof(size_t);
    memcpy(gp.data + pos,
           req_handle->opt_args_->notifMsg.c_str(),
           req_handle->opt_args_->notifMsg.size());
    pos += req_handle->opt_args_->notifMsg.size();

    memcpy(gp.data + pos, &start_addr, sizeof(uintptr_t));
    pos += sizeof(uintptr_t);
    size_t len = mem_list.size();
    memcpy(gp.data + pos, &len, sizeof(size_t));
    pos += sizeof(size_t);
    for (size_t i = 0; i < mem_list.size(); ++i) {
        memcpy(gp.data + pos, &mem_list[i], sizeof(memoryMetaResp));
        pos += sizeof(memoryMetaResp);
    }
    assert(size_t(pos) == gp.size - 1);
    assert(gp.size < MAX_PAYLOAD_SIZE);
};

readSegmentAckRequest *
gismoRpcHandler::unpackReadSegmentsAckReq(const gsmb_payload &gp) {
    // The structure is
    // REQ_FLAG | REQ_HANDLE_ID | TOTAL_DESC_COUNT | MSG_LEN | MSG | START_ADDR | LEN_OF_MEMORY_META
    // | MEMORY_META_LIST
    uintptr_t req_handle_ptr = 0;
    size_t total_desc_count = 0;
    size_t msg_len = 0;
    std::string notify_msg;
    uintptr_t start_addr = 0;
    std::vector<memoryMetaResp> *mem_list = new std::vector<memoryMetaResp>();

    int pos = 1;
    memcpy(&req_handle_ptr, gp.data + pos, sizeof(uintptr_t));
    pos += sizeof(uintptr_t);
    memcpy(&total_desc_count, gp.data + pos, sizeof(size_t));
    pos += sizeof(size_t);
    memcpy(&msg_len, gp.data + pos, sizeof(size_t));
    pos += sizeof(size_t);
    notify_msg.resize(msg_len);
    memcpy(&notify_msg[0], gp.data + pos, msg_len);
    pos += msg_len;

    memcpy(&start_addr, gp.data + pos, sizeof(uintptr_t));
    size_t len = 0;
    pos += sizeof(uintptr_t);
    memcpy(&len, gp.data + pos, sizeof(size_t));
    memoryMetaResp meta;
    pos += sizeof(size_t);
    for (size_t i = 0; i < len; ++i) {
        memcpy(&meta, gp.data + pos, sizeof(memoryMetaResp));
        mem_list->emplace_back(meta);
        pos += sizeof(memoryMetaResp);
    }
    assert(size_t(pos) == gp.size - 1);
    return new readSegmentAckRequest(
        0, "", start_addr, req_handle_ptr, total_desc_count, notify_msg.c_str(), mem_list);
};

deleteFileRequest *
gismoRpcHandler::unpackDeleteFileReq(const gsmb_payload &gp) {
    // The structure is
    // REQ_FLAG | LEN | FILE_PATH | EOF
    std::string fp = "";
    size_t pos = 1;
    size_t len = 0;
    memcpy(&len, gp.data + pos, sizeof(size_t));
    pos += sizeof(size_t);
    char *buf = new char[gp.size - 1 - sizeof(size_t)];
    memcpy(buf, gp.data + pos, gp.size - 2 - sizeof(size_t));
    fp.assign(buf);
    delete[] buf;
    return new deleteFileRequest(0, "", fp.c_str());
};

void
gismoRpcHandler::packDeleteFileReq(gsmb_payload &gp, const std::string &fp) {
    // The structure is
    // REQ_FLAG | LEN | FILE_PATH | EOF
    auto data_len = fp.size();
    gp.size = 1 + sizeof(size_t) + data_len + 1;
    gp.data = new uint8_t[gp.size];
    gp.data[0] = DELETE_FILE;
    gp.data[gp.size - 1] = EOL_FLAG;
    size_t pos = 1;
    memcpy(gp.data + pos, &data_len, sizeof(size_t));
    pos += sizeof(size_t);
    memcpy(gp.data + pos, fp.c_str(), data_len);

    pos += data_len;
    assert(pos == gp.size - 1);
    assert(gp.size < MAX_PAYLOAD_SIZE);
};

uint64_t
gismoRpcHandler::sendWithRetry(const char *dst_agent, request_type rt, const gsmb_payload &gp) {
    uint64_t req_id = 0;
    for (int retried = 0; retried < RETRY_LIMIT; ++retried) {
        req_id = gsmb_send_request(mvMessageBus_, dst_agent, gp.data, gp.size, getCurrentTime());
        if (req_id) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(SLEEP_INTERVAL));
        NIXL_DEBUG << "#" << retried << " retry to send " << std::to_string(rt) << " request to "
                   << dst_agent << " with payload size " << gp.size << " from "
                   << engine_->getAgentName();
    }
    metrics_.increaseSentReqCount(rt);
    return req_id;
}

void
gismoRpcHandler::requestMemoryFlush(const nixlGismoBackendReqH *req_handle,
                                    const uintptr_t start_addr,
                                    const std::vector<memoryMeta> &mem_list) {
    gsmb_payload gp;
    packFlushSegmentsReq(gp, req_handle, start_addr, mem_list);
    auto req_id = sendWithRetry(req_handle->remote_agent_.c_str(), FLUSH_SEGMENTS, gp);
    NIXL_DEBUG << "Sent flush request " << req_id << " to " << req_handle->remote_agent_ << " from "
               << start_addr << "(" << mem_list.at(0).len_ << ")"
               << " with payload size " << gp.size << " in " << engine_->getAgentName();
    delete[] gp.data;
}

void
gismoRpcHandler::requestMemoryLoad(const nixlGismoBackendReqH *req_handle,
                                   const uintptr_t start_addr,
                                   const std::vector<memoryMeta> &mem_list) {
    gsmb_payload gp;
    packLoadSegmentsReq(gp, req_handle, start_addr, mem_list);
    auto req_id = sendWithRetry(req_handle->remote_agent_.c_str(), LOAD_SEGMENTS, gp);
    NIXL_DEBUG << "Sent load request " << req_id << " to " << req_handle->remote_agent_
               << " with payload size " << gp.size << " in " << engine_->getAgentName();
    delete[] gp.data;
}

void
gismoRpcHandler::requestReadSegmentAck(const nixlGismoBackendReqH *req_handle,
                                       const uintptr_t start_addr,
                                       const std::vector<memoryMetaResp> &mem_list) {
    gsmb_payload gp;
    packReadSegmentsAckReq(gp, req_handle, start_addr, mem_list);
    auto req_id = sendWithRetry(req_handle->remote_agent_.c_str(), READ_SEGMENTS_ACK, gp);
    NIXL_DEBUG << "Sent read segment ack request " << req_id << " to " << req_handle->remote_agent_
               << ", start_addr " << start_addr << "(" << mem_list.at(0).len_
               << ") with payload size " << gp.size << " in " << engine_->getAgentName();
    delete[] gp.data;
}

void
gismoRpcHandler::appendNotifs(notif_list_t &new_notif_list) {
    if (new_notif_list.size() == 0) {
        return;
    }
    recvdNotifList_.push_back(new_notif_list);
    NIXL_DEBUG << "Appended " << new_notif_list.size() << " to recvd_notf_list, current size "
               << recvdNotifList_.size() << " in " << engine_->getAgentName();
}

void
gismoRpcHandler::pollNotifs(notif_list_t &notif_list) {
    NIXL_DEBUG << "Before from recvd_notf_list, current size " << recvdNotifList_.size() << " in "
               << engine_->getAgentName();
    // get all notif list
    recvdNotifList_.move(notif_list);
    if (notif_list.size() > 0) {
        NIXL_DEBUG << "Polled " << notif_list.size() << " from recvd_notf_list, current size "
                   << recvdNotifList_.size() << " in " << engine_->getAgentName();
    }
}

int
gismoRpcHandler::updateRemoteAgent(const nixlGismoBackendReqH *req_handle,
                                   const nixlMetaDesc &dms,
                                   const nixlMetaDesc &sms,
                                   const uintptr_t start_addr) {
    // send rpc to notify request done
    if (req_handle->operation_ == NIXL_READ) {
        // read one desc completed
        std::vector<memoryMetaResp> mem_list;
        mem_list.push_back(memoryMetaResp{
            .addr_ = dms.addr,
            .len_ = dms.len,
        });
        // read completed, notify peer
        requestReadSegmentAck(req_handle, start_addr, mem_list);
    } else if (req_handle->operation_ == NIXL_WRITE) {
        // write one desc completed
        std::vector<memoryMeta> mem_list;
        mem_list.push_back(memoryMeta{
            .addr_ = dms.addr,
            .len_ = dms.len,
        });
        requestMemoryLoad(req_handle, start_addr,
                          mem_list); // write completed, notify peer to load
    }
    return 0;
}

void
gismoRpcHandler::handleLoadSegment(const loadSegmentRequest *req) {
    struct timeval t_start, t_end;
    if (utils_->needRecordMetrics()) {
        gettimeofday(&t_start, nullptr);
    }

    auto dst_path = utils_->generateFileForMemory(engine_->getAgentName(), req->start_addr_);
    auto md = engine_->getBackendMD(req->start_addr_);
    if (md == nullptr) {
        NIXL_WARN << "Failed to get backend md for addr " << req->start_addr_ << " in "
                  << engine_->getAgentName() << ", abort load segment request " << req->req_id_;
        // complete this request with failure
        sendCompleteResponse(LOAD_SEGMENTS, req->req_id_);
        return;
    }
    auto read_ret = utils_->readFileSegments(dst_path, md->gpu_, req->start_addr_, req->mem_list_);
    // send response to src_agent
    std::vector<memoryMetaResp> write_mem_list;
    for (auto &m : *req->mem_list_) {
        write_mem_list.push_back(memoryMetaResp{
            .addr_ = m.addr_,
            .len_ = m.len_,
            .rslt = (uint8_t)read_ret,
        });
    }
    NIXL_DEBUG << "Read " << req->req_handle_ << " in " << engine_->getAgentName() << " completed "
               << req->req_id_;
    auto done_count = ongoingXferReqs_.update(
        req->req_handle_, [req](size_t &item) { item += req->mem_list_->size(); });
    // as request handle may be reused, we need to check done_count here
    // and decrease it accordingly
    if (done_count >= req->total_desc_len_) {
        notif_list_t notfs;
        ongoingXferReqs_.update(req->req_handle_, [req, &notfs](size_t &item) {
            if (item >= req->total_desc_len_) {
                item -= req->total_desc_len_;
                notfs.emplace_back(std::make_pair(req->src_agent_, req->notify_msg_));
            }
        });
        if (notfs.size() > 0) {
            appendNotifs(notfs);
        }
    }
    ongoingXferReqs_.removeIf(req->req_handle_, [](size_t &item) { return item == 0; });

    // send response to src_agent
    gsmb_payload read_payload;
    packLoadSegmentsResp(read_payload, req->start_addr_, write_mem_list);
    metrics_.increaseSentRespCount(LOAD_SEGMENTS);
    gsmb_send_response(mvMessageBus_,
                       req->req_id_,
                       read_payload.data,
                       read_payload.size,
                       GS_PAYLOAD_TYPE_COMPLETED_RESPONSE,
                       getCurrentTime());
    delete[] read_payload.data;
    if (utils_->needRecordMetrics()) {
        gettimeofday(&t_end, nullptr);
        perfData_.update(req->mem_list_->at(0).len_, [req, t_start, t_end](metricsItem &item) {
            item.total_count_ += req->mem_list_->size();
            item.total_time_ =
                (((t_end.tv_sec - t_start.tv_sec) * 1e6) + (t_end.tv_usec - t_start.tv_usec));
        });
    }
}

void
gismoRpcHandler::handleFlushSegment(const flushSegmentRequest *req) {
    auto dst_path = utils_->generateFileForMemory(engine_->getAgentName(), req->start_addr_);
    auto md = engine_->getBackendMD(req->start_addr_);
    if (md == nullptr) {
        NIXL_WARN << "Failed to get backend md for addr " << req->start_addr_ << " in "
                  << engine_->getAgentName() << ", abort flush segment request " << req->req_id_;
        // complete this request with failure
        sendCompleteResponse(FLUSH_SEGMENTS, req->req_id_);
        return;
    }
    auto write_ret =
        utils_->writeFileSegments(dst_path, md->gpu_, req->start_addr_, req->mem_list_);
    std::vector<memoryMetaResp> write_mem_list;
    for (auto &m : *req->mem_list_) {
        write_mem_list.push_back(memoryMetaResp{
            .addr_ = m.addr_,
            .len_ = m.len_,
            .rslt = (uint8_t)write_ret,
        });
        NIXL_DEBUG << "Flush segment" << ", start_addr " << req->start_addr_ << ", addr " << m.addr_
                   << "(" << m.len_ << ")" << "in req " << req->req_id_ << " from "
                   << req->src_agent_ << " completed at " << engine_->getAgentName();
    }
    // send response to src_agent
    gsmb_payload write_payload;
    metrics_.increaseSentRespCount(FLUSH_SEGMENTS);
    packFlushSegmentsResp(write_payload, req->req_handle_, req->start_addr_, write_mem_list);
    gsmb_send_response(mvMessageBus_,
                       req->req_id_,
                       write_payload.data,
                       write_payload.size,
                       GS_PAYLOAD_TYPE_COMPLETED_RESPONSE,
                       getCurrentTime());
    delete[] write_payload.data;
}

void
gismoRpcHandler::sendCompleteResponse(request_type req_type, const uint64_t req_id) {
    // complete this request with failure
    char buf[5] = {0};
    buf[0] = req_type;
    buf[4] = EOL_FLAG;
    metrics_.increaseSentRespCount(req_type);
    gsmb_send_response(mvMessageBus_,
                       req_id,
                       buf,
                       sizeof(buf),
                       GS_PAYLOAD_TYPE_COMPLETED_RESPONSE,
                       getCurrentTime());
}

void
gismoRpcHandler::handleReadSegmentAck(const readSegmentAckRequest *req) {
    for (auto &m : *req->mem_list_) {
        NIXL_DEBUG << "Read segment ack" << ", start_addr " << req->start_addr_ << ", addr "
                   << m.addr_ << "(" << m.len_ << ")" << "in req " << req->req_id_ << " from "
                   << req->src_agent_ << " completed at " << engine_->getAgentName();
    };
    auto done_count = ongoingXferReqs_.update(
        req->req_handle_, [req](size_t &item) { item += req->mem_list_->size(); });
    if (done_count >= req->total_desc_len_) {
        notif_list_t notfs;
        ongoingXferReqs_.update(req->req_handle_, [req, &notfs](size_t &item) {
            if (item >= req->total_desc_len_) {
                item -= req->total_desc_len_;
                notfs.emplace_back(std::make_pair(req->src_agent_, req->notify_msg_));
            }
        });
        if (notfs.size() > 0) {
            appendNotifs(notfs);
        }
    }
    ongoingXferReqs_.removeIf(req->req_handle_, [](size_t &item) { return item == 0; });
    sendCompleteResponse(READ_SEGMENTS_ACK, req->req_id_);
}

void
gismoRpcHandler::handleDeleteFile(const deleteFileRequest *req) {
    utils_->deleteFileCache(req->filePath.c_str());
    sendCompleteResponse(DELETE_FILE, req->req_id_);
}

void
gismoRpcHandler::handleLoadSegmentResp(const loadSegmentResponse *rsp) {
    for (auto &m : *rsp->mem_list_) {
        NIXL_DEBUG << "memory loaded from " << rsp->src_agent_ << ", addr " << m.addr_ << " in "
                   << engine_->getAgentName();
    };
}

void
gismoRpcHandler::handleFlushSegmentResp(const flushSegmentResponse *rsp) {
    for (auto &m : *rsp->mem_list_) {
        NIXL_DEBUG << "append flush notify from " << rsp->src_agent_ << ", start_addr "
                   << rsp->start_addr_ << ", addr " << m.addr_ << " in " << engine_->getAgentName();
        // update the time of segment flush
        utils_->getReqMgr()->setDescUpdateTime(rsp->req_handle_, m.addr_, getCurrentTime());
    };

    NIXL_DEBUG << "Notified flush segment resp " << rsp->rsp_id_ << " from " << rsp->src_agent_
               << ", start_addr " << rsp->start_addr_ << ", segments " << rsp->mem_list_->size()
               << " in " << engine_->getAgentName();
}

int
gismoRpcHandler::broadcastRequest(const gsmb_payload &pl, request_type rt) {
    std::vector<std::string> agents;
    auto ret = utils_->listAllAgents(agents);
    if (ret != 0) {
        return ret;
    }
    for (auto &ag : agents) {
        if (ag == engine_->getAgentName()) {
            continue;
        }
        auto req_id = sendWithRetry(ag.c_str(), rt, pl);
        NIXL_DEBUG << "Sent req " << req_id << " to " << ag << " in " << engine_->getAgentName();
    }
    return 0;
}

int
gismoRpcHandler::broadcastFileDelete(const char *fp) {
    NIXL_DEBUG << "Broadcast nodes to delete " << fp << " in " << engine_->getAgentName();
    gsmb_payload pl;
    packDeleteFileReq(pl, fp);
    auto ret = broadcastRequest(pl, DELETE_FILE);
    delete[] pl.data;
    return ret;
}