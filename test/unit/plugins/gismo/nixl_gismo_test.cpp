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
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <cassert>
#include <cstring>
#include <thread>
#include <absl/strings/str_format.h>

#include "gismo_backend.h"
#include "gismo_utils.h"

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cuda.h>

int gpu_id = 0;

static void
checkCudaError(cudaError_t result, const char *message) {
    if (result != cudaSuccess) {
        std::cerr << message << " (Error code: " << result << " - " << cudaGetErrorString(result)
                  << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif

using namespace std;

class testHndlIterator {
private:
    bool reuse_;
    bool set_;
    bool prepare_;
    bool release_;
    nixlBackendReqH *handle_;

public:
    testHndlIterator(bool reuse) {
        reuse_ = reuse;
        if (reuse) {
            prepare_ = true;
            release_ = false;
        } else {
            prepare_ = true;
            release_ = true;
        }
        handle_ = nullptr;
        set_ = false;
    }

    ~testHndlIterator() {
        /* Make sure that handler was released */
        assert(!set_);
    }

    bool
    needPrep() {
        if (reuse_) {
            if (!prepare_) {
                return false;
            }
        }
        return true;
    }

    bool
    needRelease() {
        return release_;
    }

    void
    isLast() {
        if (reuse_) {
            release_ = true;
        }
    }

    void
    setHandle(nixlBackendReqH *handle) {
        assert(!set_);
        handle_ = handle;
        set_ = true;
        if (reuse_) {
            prepare_ = false;
        }
    }

    void
    unsetHandle() {
        assert(set_);
        set_ = false;
    }

    nixlBackendReqH *&
    getHandle() {
        assert(set_);
        return handle_;
    }
};

nixlBackendEngine *
createEngine(std::string name, std::string socket_path, bool p_thread, bool use_mmap) {
    nixlBackendEngine *gismo;
    nixlBackendInitParams init;
    nixl_b_params_t custom_params;

    init.enableProgTh = p_thread;
    init.pthrDelay = 100;
    init.localAgent = name;
    if (socket_path.empty()) {
        auto env = getenv("DMO_SOCKET_PATH");
        if (env) {
            socket_path = env;
        }
    }
    if (socket_path.empty()) {
        socket_path = "dmo.daemon.sock.sheperd";
    }
    custom_params["socket_path"] = socket_path;
    custom_params["use_mmap"] = use_mmap ? "true" : "false";
    custom_params["chunk_size"] =
        use_mmap ? std::to_string(1024 * 1024 * 2) : std::to_string(1024 * 256);
    init.customParams = &custom_params;
    init.type = "gismo";

    gismo = (nixlBackendEngine *)new nixlGismoEngine(&init);
    assert(!gismo->getInitErr());
    if (gismo->getInitErr()) {
        std::cout << "Failed to initialize worker1" << std::endl;
        exit(1);
    }

    return gismo;
}

void
releaseEngine(nixlBackendEngine *gismo) {
    delete gismo;
}

std::string
memType2Str(nixl_mem_t mem_type) {
    switch (mem_type) {
    case DRAM_SEG:
        return std::string("DRAM");
    case VRAM_SEG:
        return std::string("VRAM");
    case BLK_SEG:
        return std::string("BLOCK");
    case FILE_SEG:
        return std::string("FILE");
    default:
        std::cout << "Unsupported memory type!" << std::endl;
        assert(0);
    }
}

void
allocateBuffer(nixl_mem_t mem_type, int dev_id, size_t len, void *&addr) {
    switch (mem_type) {
    case DRAM_SEG:
        addr = calloc(1, len);
        break;
    default:
        std::cout << "Unsupported memory type!" << std::endl;
        assert(0);
    }
    assert(addr);
}

void
releaseBuffer(nixl_mem_t mem_type, int dev_id, void *&addr) {
    switch (mem_type) {
    case DRAM_SEG:
        free(addr);
        break;
    default:
        std::cout << "Unsupported memory type!" << std::endl;
        assert(0);
    }
}

void
doMemset(nixl_mem_t mem_type, int dev_id, void *addr, char byte, size_t len) {
    switch (mem_type) {
    case DRAM_SEG:
        memset(addr, byte, len);
        break;
    default:
        std::cout << "Unsupported memory type!" << std::endl;
        assert(0);
    }
}

void *
getValidationPtr(nixl_mem_t mem_type, void *addr, size_t len) {
    switch (mem_type) {
    case DRAM_SEG:
        return addr;
        break;
    default:
        cout << "Unsupported memory type!" << endl;
        assert(0);
    }
    return NULL;
}

void *
releaseValidationPtr(nixl_mem_t mem_type, void *addr) {
    switch (mem_type) {
    case DRAM_SEG:
        break;
    default:
        cout << "Unsupported memory type!" << endl;
        assert(0);
    }
    return NULL;
}

void
allocateWrongGPUTest(nixlBackendEngine *gismo, int dev_id) {
    nixlBlobDesc desc;
    nixlBackendMD *md;
    void *buf;

    allocateBuffer(VRAM_SEG, dev_id, desc.len, buf);

    desc.devId = dev_id;
    desc.addr = (uint64_t)buf;

    int ret = gismo->registerMem(desc, VRAM_SEG, md);

    assert(ret == NIXL_ERR_NOT_SUPPORTED);

    releaseBuffer(VRAM_SEG, dev_id, buf);
}

void
allocateAndRegister(nixlBackendEngine *gismo,
                    int dev_id,
                    nixl_mem_t mem_type,
                    void *&addr,
                    size_t len,
                    nixlBackendMD *&md) {
    nixlBlobDesc desc;

    allocateBuffer(mem_type, dev_id, len, addr);

    desc.addr = (uintptr_t)addr;
    desc.len = len;
    desc.devId = dev_id;

    int ret = gismo->registerMem(desc, mem_type, md);

    assert(ret == NIXL_SUCCESS);
}

void
deallocateAndDeregister(nixlBackendEngine *gismo,
                        int dev_id,
                        nixl_mem_t mem_type,
                        void *&addr,
                        nixlBackendMD *&md) {
    gismo->deregisterMem(md);
    releaseBuffer(mem_type, dev_id, addr);
}

void
loadRemote(nixlBackendEngine *gismo,
           int dev_id,
           std::string agent,
           nixl_mem_t mem_type,
           void *addr,
           size_t len,
           nixlBackendMD *&lmd,
           nixlBackendMD *&rmd) {
    nixlBlobDesc info;
    info.addr = (uintptr_t)addr;
    info.len = len;
    info.devId = dev_id;
    gismo->getPublicData(lmd, info.metaInfo);

    // We get the data from the cetnral location and populate the backend, and receive remote_meta
    int ret = gismo->loadRemoteMD(info, mem_type, agent, rmd);
    assert(NIXL_SUCCESS == ret);
}

void
populateDescs(nixl_meta_dlist_t &descs,
              int dev_id,
              void *addr,
              int desc_cnt,
              size_t desc_size,
              nixlBackendMD *&md) {
    for (int i = 0; i < desc_cnt; i++) {
        nixlMetaDesc req;
        req.addr = (uintptr_t)(((char *)addr) + i * desc_size); // random offset
        req.len = desc_size;
        req.devId = dev_id;
        req.metadataP = md;
        descs.addDesc(req);
    }
}

static string
op2string(nixl_xfer_op_t op, bool has_notif) {
    if (op == NIXL_READ && !has_notif) return string("READ");
    if (op == NIXL_WRITE && !has_notif) return string("WRITE");
    if (op == NIXL_READ && has_notif) return string("READ/NOTIF");
    if (op == NIXL_WRITE && has_notif) return string("WRITE/NOTIF");

    return string("ERR-OP");
}

void
performTransfer(nixlBackendEngine *x1,
                nixlBackendEngine *x2,
                nixl_meta_dlist_t &req_src_descs,
                nixl_meta_dlist_t &req_dst_descs,
                void *addr1,
                void *addr2,
                int iteration,
                size_t len,
                nixl_xfer_op_t op,
                testHndlIterator &hiter,
                bool use_notif) {
    int ret2;
    nixl_status_t ret3;
    void *chkptr1, *chkptr2;
    auto gismo_eng1 = dynamic_cast<nixlGismoEngine *>(x1);
    auto gismo_eng2 = dynamic_cast<nixlGismoEngine *>(x2);

    std::string local_agent = gismo_eng1->getAgentName();
    std::string remote_agent = gismo_eng2->getAgentName();

    std::string test_str("test-iteration-" + to_string(iteration));
    std::cout << "\t" << op2string(op, use_notif) << " from " << local_agent << "("
              << (uintptr_t)addr1 << ") to " << remote_agent << "(" << (uintptr_t)addr2 << ") len "
              << len << " iteration " << iteration << "\n";

    nixl_opt_b_args_t opt_args;
    opt_args.notifMsg = test_str;
    opt_args.hasNotif = use_notif;

    // Posting a request, to be updated to return an async handler,
    // or an ID that later can be used to check the status as a new method
    // Also maybe we would remove the WRITE and let the backend class decide the op
    if (hiter.needPrep()) {
        nixlBackendReqH *new_handle = nullptr;
        ret3 = x1->prepXfer(op, req_src_descs, req_dst_descs, remote_agent, new_handle, &opt_args);
        assert(ret3 == NIXL_SUCCESS);
        hiter.setHandle(new_handle);
    }
    nixlBackendReqH *&handle = hiter.getHandle();
    ret3 = x1->prepXfer(op, req_src_descs, req_dst_descs, remote_agent, handle, &opt_args);
    assert(ret3 == NIXL_SUCCESS);
    nixlGismoBackendReqH *gismo_handle = dynamic_cast<nixlGismoBackendReqH *>(handle);
    assert(gismo_handle != nullptr);
    gismo_handle->opt_args_->notifMsg = test_str; // update the notif msg in case of reuse
    ret3 = x1->postXfer(op, req_src_descs, req_dst_descs, remote_agent, handle, &opt_args);
    assert(ret3 == NIXL_SUCCESS || ret3 == NIXL_IN_PROG);

    if (ret3 == NIXL_SUCCESS) {
        cout << "\t\tWARNING: Tansfer request completed immediately - no testing non-inline path"
             << endl;
    } else {
        cout << "\t\tNOTE: Testing non-inline Transfer path!" << endl;

        while (ret3 == NIXL_IN_PROG) {
            ret3 = x1->checkXfer(handle);
            assert(ret3 == NIXL_SUCCESS || ret3 == NIXL_IN_PROG);
        }
    }
    if (hiter.needRelease()) {
        hiter.unsetHandle();
        x1->releaseReqH(handle);
    }

    if (use_notif) {
        /* Test notification path */
        notif_list_t target_notifs;

        cout << "\t\tChecking notification flow: " << flush;
        ret2 = 0;

        while (ret2 == 0) {
            ret3 = x2->getNotifs(target_notifs);
            ret2 = target_notifs.size();
            assert(ret3 == NIXL_SUCCESS);
        }

        assert(ret2 >= 1);
        assert(target_notifs.front().first == local_agent);
        assert(target_notifs.front().second == test_str);

        cout << "OK" << endl;
    }

    cout << "\t\tData verification: " << flush;

    chkptr1 = getValidationPtr(req_src_descs.getType(), addr1, len);
    chkptr2 = getValidationPtr(req_dst_descs.getType(), addr2, len);

    cout << "Ready to check memory content between " << (uintptr_t)addr1 << " and "
         << (uintptr_t)addr2 << endl;
    // Perform correctness check.
    for (size_t i = 0; i < len; i++) {
        if (((uint8_t *)chkptr1)[i] != ((uint8_t *)chkptr2)[i]) {
            cout << "\t\tChecking " << i << " failed. Expect " << (int)((uint8_t *)chkptr1)[i]
                 << " but got " << (int)((uint8_t *)chkptr2)[i] << endl;
        }
        assert(((uint8_t *)chkptr1)[i] == ((uint8_t *)chkptr2)[i]);
    }

    releaseValidationPtr(req_src_descs.getType(), chkptr1);
    releaseValidationPtr(req_dst_descs.getType(), chkptr2);

    cout << "\tOK" << endl;
}

void
test_intra_agent_transfer(int round, nixlBackendEngine *xw, nixl_mem_t mem_type) {

    std::cout << std::endl << std::endl;
    std::cout << "****************************************************" << std::endl;
    std::cout << "   Intra-agent memory transfer (offload) test: " << std::endl;
    std::cout << "       Round=" << round << ", " << memType2Str(mem_type) << std::endl;
    std::cout << "****************************************************" << std::endl;
    std::cout << std::endl << std::endl;

    auto gismo_eng1 = dynamic_cast<nixlGismoEngine *>(xw);
    std::string local_agent = gismo_eng1->getAgentName();
    nixl_status_t ret1;

    int iter = 10;

    assert(xw->supportsLocal());

    // connection info is still a string
    std::string conn_info1;
    ret1 = xw->getConnInfo(conn_info1);
    assert(ret1 == NIXL_SUCCESS);
    ret1 = xw->loadRemoteConnInfo(local_agent, conn_info1);
    assert(ret1 == NIXL_SUCCESS);

    std::cout << "Local connection complete\n";

    // Number of transfer descriptors
    int desc_cnt = 64;
    // Size of a single descriptor
    size_t desc_size = 1 * 1024 * 1024;
    size_t len = desc_cnt * desc_size;

    void *addr1, *addr2;
    nixlBackendMD *lmd1, *lmd2;
    allocateAndRegister(xw, 0, mem_type, addr1, len, lmd1);
    allocateAndRegister(xw, 0, mem_type, addr2, len, lmd2);

    // string descs unnecessary, convert meta locally
    nixlBackendMD *rmd2;
    ret1 = xw->loadLocalMD(lmd2, rmd2);
    assert(ret1 == NIXL_SUCCESS);

    nixl_meta_dlist_t req_src_descs(mem_type);
    populateDescs(req_src_descs, 0, addr1, desc_cnt, desc_size, lmd1);

    nixl_meta_dlist_t req_dst_descs(mem_type);
    populateDescs(req_dst_descs, 0, addr2, desc_cnt, desc_size, rmd2);

    nixl_xfer_op_t ops[] = {NIXL_READ, NIXL_WRITE};
    bool use_notifs[] = {false}; // gismo transfer engine doesn't support notifs

    for (size_t i = 0; i < sizeof(ops) / sizeof(ops[i]); i++) {

        for (bool use_notif : use_notifs) {
            cout << endl
                 << op2string(ops[i], use_notif) << " test (" << iter << ") iterations" << endl;
            for (int k = 0; k < iter; k++) {
                /* Init data */
                doMemset(mem_type, 0, addr1, 0xbb, len);
                doMemset(mem_type, 0, addr2, 0, len);

                /* Test */
                testHndlIterator hiter(false);
                performTransfer(xw,
                                xw,
                                req_src_descs,
                                req_dst_descs,
                                addr1,
                                addr2,
                                k,
                                len,
                                ops[i],
                                hiter,
                                use_notif);
            }
        }
    }

    xw->unloadMD(rmd2);
    deallocateAndDeregister(xw, 0, mem_type, addr1, lmd1);
    deallocateAndDeregister(xw, 0, mem_type, addr2, lmd2);

    xw->disconnect(local_agent);
}

void
test_inter_agent_transfer(int round,
                          bool reuse_hndl,
                          nixlBackendEngine *x1,
                          nixl_mem_t src_mem_type,
                          int src_dev_id,
                          nixlBackendEngine *x2,
                          nixl_mem_t dst_mem_type,
                          int dst_dev_id) {
    int ret;
    int iter = 10;

    std::cout << std::endl << std::endl;
    std::cout << "****************************************************" << std::endl;
    std::cout << "    Inter-agent memory transfer test " << std::endl;
    std::cout << "         Round=" << round << std::endl;
    std::cout << "         ReqHandle-reuse=" << (reuse_hndl ? "ON" : "OFF") << std::endl;
    std::cout << "         (" << memType2Str(src_mem_type) << " -> " << memType2Str(dst_mem_type)
              << ")" << std::endl;
    std::cout << "****************************************************" << std::endl;
    std::cout << std::endl << std::endl;

    // Example: assuming two agents running on the same machine,
    // with separate memory regions in DRAM

    auto gismo_eng1 = dynamic_cast<nixlGismoEngine *>(x1);
    auto gismo_eng2 = dynamic_cast<nixlGismoEngine *>(x2);
    std::string agent1 = gismo_eng1->getAgentName();
    std::string agent2 = gismo_eng2->getAgentName();

    // We get the required connection info from gismo to be put on the central
    // location and ask for it for a remote node
    std::string conn_info1, conn_info2;
    ret = x1->getConnInfo(conn_info1);
    assert(ret == NIXL_SUCCESS);
    ret = x2->getConnInfo(conn_info2);
    assert(ret == NIXL_SUCCESS);

    // We assumed we put them to central location and now receiving it on the other process
    ret = x1->loadRemoteConnInfo(agent2, conn_info2);
    assert(ret == NIXL_SUCCESS);

    std::cout << "Synchronous handshake complete\n";

    // Number of transfer descriptors
    int desc_cnt = 64;
    // Size of a single descriptor
    size_t desc_size = 1 * 1024 * 1024;
    size_t len = desc_cnt * desc_size;

    void *addr1 = NULL, *addr2 = NULL;
    nixlBackendMD *lmd1, *lmd2;
    allocateAndRegister(x1, src_dev_id, src_mem_type, addr1, len, lmd1);
    allocateAndRegister(x2, dst_dev_id, dst_mem_type, addr2, len, lmd2);

    nixlBackendMD *rmd1 /*, *rmd2*/;
    loadRemote(x1, dst_dev_id, agent2, dst_mem_type, addr2, len, lmd2, rmd1);

    nixl_meta_dlist_t req_src_descs(src_mem_type);
    populateDescs(req_src_descs, src_dev_id, addr1, desc_cnt, desc_size, lmd1);

    nixl_meta_dlist_t req_dst_descs(dst_mem_type);
    populateDescs(req_dst_descs, dst_dev_id, addr2, desc_cnt, desc_size, rmd1);

    nixl_xfer_op_t ops[] = {NIXL_WRITE, NIXL_READ};
    bool use_notifs[] = {true}; // gismo transfer engine support notifs

    for (size_t i = 0; i < sizeof(ops) / sizeof(ops[i]); i++) {

        for (bool use_notif : use_notifs) {
            cout << endl
                 << op2string(ops[i], use_notif) << " test (" << iter << ") iterations" << endl;
            testHndlIterator hiter(reuse_hndl);
            for (int k = 0; k < iter; k++) {
                /* Init data */
                doMemset(src_mem_type, src_dev_id, addr1, 0xbb, len);
                doMemset(dst_mem_type, dst_dev_id, addr2, 0xda, len);

                /* Test */
                if ((k + 1) == iter) {
                    /* If this is the last iteration */
                    hiter.isLast();
                }
                performTransfer(x1,
                                x2,
                                req_src_descs,
                                req_dst_descs,
                                addr1,
                                addr2,
                                k,
                                len,
                                ops[i],
                                hiter,
                                use_notif);
            }
        }
    }

    // As well as all the remote notes, asking to remove them one by one
    // need to provide list of descs
    x1->unloadMD(rmd1);


    // Release memory regions
    deallocateAndDeregister(x1, src_dev_id, src_mem_type, addr1, lmd1);
    deallocateAndDeregister(x2, dst_dev_id, dst_mem_type, addr2, lmd2);

    // Test one-sided disconnect (initiator only)
    x1->disconnect(agent2);
}

int
doGismoTests(std::string socket_path, bool use_mmap) {
    bool thread_on[2] = {false, true};
    nixlBackendEngine *gismo[2][2] = {0};

    // Allocate gismo engines
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            auto agent_name = std::string("Agent-") + std::to_string(i) + "-" + std::to_string(j);
            gismo[i][j] = createEngine(agent_name.c_str(), socket_path, thread_on[i], use_mmap);
        }
    }

#ifdef HAVE_CUDA
    int dev_ids[2] = {0, 0};
    int n_vram_dev;
    if (cudaGetDeviceCount(&n_vram_dev) != cudaSuccess) {
        std::cout << "Call to cudaGetDeviceCount failed, assuming 0 devices";
        n_vram_dev = 0;
    }

    std::cout << "Detected " << n_vram_dev << " CUDA devices" << std::endl;
    if (n_vram_dev > 1) {
        dev_ids[1] = 1;
        dev_ids[0] = 0;
    }
#endif

    for (int i = 0; i < 2; i++) {
        // Test local memory to local memory transfer
        test_intra_agent_transfer(i, gismo[i][0], DRAM_SEG);
#ifdef HAVE_CUDA
        if (n_vram_dev > 0) {
            test_intra_agent_transfer(i, gismo[i][0], VRAM_SEG);
        }
#endif
    }

    for (int i = 0; i < 2; i++) {
        test_inter_agent_transfer(i, false, gismo[i][0], DRAM_SEG, 0, gismo[i][1], DRAM_SEG, 0);
        test_inter_agent_transfer(i, true, gismo[i][0], DRAM_SEG, 0, gismo[i][1], DRAM_SEG, 0);

#ifdef HAVE_CUDA
        if (n_vram_dev > 1) {
            test_inter_agent_transfer(
                i, false, gismo[i][0], VRAM_SEG, dev_ids[0], gismo[i][1], VRAM_SEG, dev_ids[1]);
            test_inter_agent_transfer(
                i, true, gismo[i][0], VRAM_SEG, dev_ids[0], gismo[i][1], VRAM_SEG, dev_ids[1]);
            test_inter_agent_transfer(
                i, true, gismo[i][0], DRAM_SEG, dev_ids[0], gismo[i][1], VRAM_SEG, dev_ids[1]);
            test_inter_agent_transfer(
                i, true, gismo[i][0], VRAM_SEG, dev_ids[0], gismo[i][1], DRAM_SEG, dev_ids[1]);
        }
#endif
    }

#ifdef HAVE_CUDA
    if (n_vram_dev > 1) {
        // Test if registering on a different GPU fails correctly
        allocateWrongGPUTest(gismo[0][0], 1);
        std::cout << "Verified registration on wrong GPU fails correctly\n";
    }
#endif

    // Deallocate gismo engines
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            releaseEngine(gismo[i][j]);
        }
    }
    return 0;
}

int
main(int argc, char **argv) {
    // accept socket path as an optional argument
    // if not provided, default to DMO_SOCKET_PATH env variable or "dmo.daemon.sock.sheperd"
    int opt;
    std::string socket_path = "";
    while ((opt = getopt(argc, argv, "hp:")) != -1) {
        switch (opt) {
        case 'p':
            socket_path = optarg;
            break;
        case 'h':
        case '?':
            std::cout << absl::StrFormat("Usage: %s [-p socket_path] or use DMO_SOCKET_PATH env variable", argv[0]) << std::endl;
            return 0;
        }
    }
    if (socket_path.empty()) {
        std::cout
            << "No socket path provided, will use DMO_SOCKET_PATH env variable or default value"
            << std::endl;
    }

    std::cout << "****************************************************" << std::endl;
    std::cout << "    Testing with mmap optimization enabled" << std::endl;
    std::cout << "****************************************************" << std::endl;
    doGismoTests(socket_path, true);
    std::cout << std::endl << std::endl;
    std::cout << "****************************************************" << std::endl;
    std::cout << "    Testing with mmap optimization disabled" << std::endl;
    std::cout << "****************************************************" << std::endl;
    std::cout << std::endl << std::endl;
    doGismoTests(socket_path, false);
    return 0;
}