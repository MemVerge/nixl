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

#include <cstdint>
#include <cstring>
#include <cerrno>
#include <filesystem>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "backend_aux.h"
#include "common/nixl_log.h"
#include "common/str_tools.h"
#include "gismo_utils.h"
#include "gismo_log.h"
#include "gismo_rpc_handler.h"
#include "gismo_semaphore.h"
#include "mvfs.h"
#include "nixl_types.h"

// serialize this object to a string
std::string
nixlGismoBackendMD::toString() {
    char buf[256] = {0};
    snprintf(buf, 255, "%lu;%lu;%d;%d", (uintptr_t)addr_, length_, ref_cnt_, gpu_);
    return std::string(buf);
}

// deserialize from a string
void
nixlGismoBackendMD::fromString(std::string &objInStr) {
    auto fields = str_split(objInStr, ";");
    if (fields.size() < 4) {
        return;
    }
    this->addr_ = strToInt(fields.at(0));
    this->length_ = strToInt(fields.at(1));
    this->ref_cnt_ = strToInt(fields.at(2));
    this->gpu_ = strToInt(fields[3]);
}

gismoUtils::gismoUtils(const char *id) : connection_(nullptr), localAgentId_(id) {
    reqMgr_ = new gismoReqMgr();
    int device_count = 0;
    cudaError_t error_id = cudaGetDeviceCount(&device_count);

    if (error_id != cudaSuccess) {
        NIXL_INFO << cudaGetErrorString(error_id)
                  << " -> No CUDA device found or CUDA driver not installed correctly.";
        return;
    }
    hasCudaDevices_ = device_count > 0;
}

gismoUtils::~gismoUtils() {
    if (threadPool_) {
        delete threadPool_;
        threadPool_ = nullptr;
    }
    if (reqMgr_) {
        delete reqMgr_;
        reqMgr_ = nullptr;
    }
    std::vector<std::string> keys;
    remoteMemInfo_.keys(keys);
    for (auto &k : keys) {
        auto p = remoteMemInfo_.get(k);
        assert(p);
        for (auto md : p.value().second) {
            delete md;
        }
        remoteMemInfo_.remove(k);
    }

    deinitMVFS();
}

int
gismoUtils::initConfigParams(const nixlBackendInitParams *p) {
    if (p->customParams != NULL) {
        auto sp = p->customParams->find(SOCKET_PATH);
        if (sp != p->customParams->end() && sp->second.length() > 0) {
            configParams_.socket_path_ = sp->second;
        } else {
            if (!std::filesystem::exists(configParams_.socket_path_)) {
                NIXL_WARN << "Default socket path " << configParams_.socket_path_
                          << " does not exist, try the one under /var/run/dmo.";
                configParams_.socket_path_ = "/var/run/dmo/dmo.daemon.sock.0";
                if (!std::filesystem::exists(configParams_.socket_path_)) {
                    NIXL_WARN << "Socket path " << configParams_.socket_path_
                              << " does not exist, try environment variable GISMO_SOCKET_PATH.";
                    auto sock_env = std::getenv(SOCKET_PATH_ENV);
                    if (sock_env) {
                        configParams_.socket_path_ = sock_env;
                    } else {
                        NIXL_ERROR << "Environment variable GISMO_SOCKET_PATH is not set, cannot "
                                      "find valid socket path.";
                        return NIXL_ERR_BACKEND;
                    }
                }
            }
        }

        sp = p->customParams->find(CLIENT_LOG_PATH);
        if (sp != p->customParams->end()) {
            configParams_.client_log_path_ = sp->second;
        } else {
            NIXL_INFO << "Not found client_log_path in init params, use default "
                      << configParams_.client_log_path_;
        }

        sp = p->customParams->find(THREAD_POOL_SIZE);
        if (sp != p->customParams->end()) {
            configParams_.thread_pool_size_ = strToInt(sp->second);
        } else {
            auto env = std::getenv(THREAD_POOL_SIZE_ENV);
            if (env != nullptr) {
                configParams_.thread_pool_size_ = strToInt(env);
            }
        }

        sp = p->customParams->find(RECORD_METRICS);
        if (sp != p->customParams->end()) {
            configParams_.record_metrics_ = sp->second == "true";
        }

        sp = p->customParams->find(USE_MMAP);
        if (sp != p->customParams->end()) {
            configParams_.use_mmap_ = sp->second == "true";
        } else {
            auto env = std::getenv(USE_MMAP_ENV);
            if (env != nullptr) {
                configParams_.use_mmap_ = std::string(env) == "true";
            }
        }

        sp = p->customParams->find(CHUNK_SIZE);
        if (sp != p->customParams->end()) {
            configParams_.chunk_size_ = strToInt(sp->second);
        } else {
            auto env = std::getenv(CHUNK_SIZE_ENV);
            if (env != nullptr) {
                configParams_.chunk_size_ = strToInt(env);
            }
        }
    } else {
        auto env = std::getenv(CHUNK_SIZE_ENV);
        if (env != nullptr) {
            configParams_.chunk_size_ = strToInt(env);
        }
        env = std::getenv(USE_MMAP_ENV);
        if (env != nullptr) {
            configParams_.use_mmap_ = std::string(env) == "true";
        }
        env = std::getenv(THREAD_POOL_SIZE_ENV);
        if (env != nullptr) {
            configParams_.thread_pool_size_ = strToInt(env);
        }
        env = std::getenv(SOCKET_PATH_ENV);
        if (env) {
            configParams_.socket_path_ = env;
        }
    }
    if (configParams_.chunk_size_ < 4096) {
        configParams_.chunk_size_ = DEFAULT_CHUNK_SIZE;
    }
    NIXL_WARN << "Using socket_path " << configParams_.socket_path_ << " in Gismo backend "
              << localAgentId_ << ".";
    NIXL_WARN << "Using chunk_size " << configParams_.chunk_size_ << " in Gismo backend "
              << localAgentId_ << ".";
    NIXL_WARN << "Using use_mmap " << (configParams_.use_mmap_ ? "true" : "false")
              << " in Gismo backend " << localAgentId_ << ".";
    NIXL_WARN << "Using thread_pool_size " << configParams_.thread_pool_size_
              << " in Gismo backend " << localAgentId_ << ".";
    return 0;
}

int
gismoUtils::initMVFS(const nixlBackendInitParams *p) {
    auto rc = initConfigParams(p);
    if (rc != 0) {
        return rc;
    }
    mvfs_connect_option conn_option;
    mvfs_register_temp_folder_option temp_folder_options;

    mvfs_init_log(configParams_.client_log_path_.c_str(), 10, 3, "Info"); // Critical| Warning
    strncpy(conn_option.socket, configParams_.socket_path_.c_str(), sizeof(conn_option.socket) - 1);
    conn_option.socket[sizeof(conn_option.socket) - 1] = '\0';
    conn_option.warmup = false;
    std::string dir = ROOT_DIR + p->localAgent;
    strncpy(temp_folder_options.folders, dir.c_str(), sizeof(temp_folder_options.folders) - 1);
    temp_folder_options.folders[sizeof(temp_folder_options.folders) - 1] = '\0';
    auto ret = mvfs_connect_with_temp_folders(&connection_, &conn_option, &temp_folder_options);
    if (ret) {
        NIXL_ERROR << "Failed to connect DMO using socket " << configParams_.socket_path_
                   << ", error " << errno;
        return NIXL_ERR_BACKEND;
    }
    rc = mvfs_mkdir_mc(connection_, dir.c_str(), ACCESSPERMS);
    if (rc != 0) {
        NIXL_ERROR << "Failed to create " << dir << " in DMO, ret " << rc << ", errno " << errno;
        mvfs_disconnect_mc(connection_);
        connection_ = nullptr;
        return NIXL_ERR_BACKEND;
    }
    rc = mvfs_open_gsmb_mc(connection_, "MemVerge-GISMO", localAgentId_, &mvMessageBus_);
    if (rc != 0) {
        NIXL_ERROR << "Failed to create message bus " << localAgentId_ << ", error " << rc;
        mvfs_rmdir_mc(connection_, dir.c_str());
        mvfs_disconnect_mc(connection_);
        connection_ = nullptr;
        return NIXL_ERR_BACKEND;
    }
    threadPool_ = new gismoThreadPool(localAgentId_, configParams_.thread_pool_size_);
    return NIXL_SUCCESS;
}

void
gismoUtils::deinitMVFS() {
    // stop threadpool first
    if (threadPool_) {
        delete threadPool_;
        threadPool_ = nullptr;
    }
    // clean cached fd
    std::vector<std::string> file_list;
    openedMvfsFds_.keys(file_list);
    for (auto &f : file_list) {
        deleteFileCache(f.c_str());
    }

    if (mvMessageBus_) {
        mvfs_close_gsmb_mc(connection_, mvMessageBus_);
        mvMessageBus_ = nullptr;
    }
    if (connection_) {
        std::string dir = std::string(ROOT_DIR) + localAgentId_;
        auto rc = mvfs_rmdir_mc(connection_, dir.c_str());
        if (rc != 0) {
            NIXL_WARN << "Failed to delete " << dir << ", error " << rc;
        }
        mvfs_disconnect_mc(connection_);
        connection_ = nullptr;
    }
    if (configParams_.record_metrics_) {
        std::vector<uint32_t> keys;
        perfData_.keys(keys);
        for (auto &k : keys) {
            auto item = perfData_.get(k);
            assert(item);
            NIXL_WARN << "chunk_size: " << k << ", total_count: " << item.value().total_count_
                      << ", total_time: " << item.value().total_time_
                      << ", total_write_time: " << item.value().total_write_time_
                      << ", total_notify_time: "
                      << item.value().total_time_ - item.value().total_write_time_
                      << ", avg_time: " << item.value().total_time_ / item.value().total_count_
                      << "us"
                      << ", avg_write_time: "
                      << item.value().total_write_time_ / item.value().total_count_ << "us"
                      << ", avg_notify_time: "
                      << (item.value().total_time_ - item.value().total_write_time_) /
                    item.value().total_count_
                      << "us";
            perfData_.remove(k);
        }
    }
}

int
gismoUtils::registerLocalMemory(bool gpu, uintptr_t addr, size_t size, nixlGismoBackendMD *&md) {
    // when register memory
    // we create one file in DMO
    std::unique_lock<std::shared_mutex> guard(memoryMtx_);
    auto fpath = generateFileForMemory(localAgentId_, addr);
    auto ret = allocFile(fpath, size);
    if (ret < 0) {
        NIXL_ERROR << "Failed to write " << fpath << " in " << localAgentId_ << ", error " << ret;
        return ret;
    }
    auto status = std::string(ROOT_DIR) + localAgentId_ + "/" + MEM_REGISTER_INFO;
    md = new nixlGismoBackendMD(true);
    md->addr_ = addr;
    md->length_ = size;
    md->ref_cnt_ = 1;
    md->gpu_ = gpu;
    auto content = md->toString();
    content += "\n";
    ret = appendFile(status.c_str(), content.c_str(), content.size());
    if (ret < 0) {
        return ret;
    }
    return 0;
}

void
gismoUtils::removeLineFromFile(const char *fp, const char *line_starter) {
    // clean the record in status file
    mvfs_handle fd;
    mvfs_file_attr attr;
    auto ret = mvfs_open_with_attr_mc(connection_, &fd, fp, O_RDWR, &attr);
    if (ret) {
        NIXL_WARN << "Failed to open " << fp << " in " << localAgentId_ << ", error " << ret;
        return;
    }
    char *buf = new char[attr.size + 1];
    ret = mvfs_read_mc(connection_, fd, buf, attr.size, 0);
    auto lines = str_split(buf, "\n");
    int pos = 0;

    mvfs_ftruncate_mc(connection_, fd, 0); // first truncate it to empty;
    for (auto &i : lines) {
        if (i.find(line_starter) == 0) {
            NIXL_DEBUG << "deleted line " << i << " from " << fp;
            continue;
        };
        strcpy(buf + pos, i.c_str());
        pos += i.length();
        buf[pos + 1] = '\n';
        pos++;
    }
    // only write back when there is content
    if (pos > 0) {
        ret = mvfs_write_mc(connection_, fd, buf, pos + 1, 0);
        if (ret < 0) {
            NIXL_WARN << "Failed to write " << fp << " in " << localAgentId_ << ", error " << ret
                      << ", errno " << errno;
        }
    }

    delete[] buf;
    mvfs_close_mc(connection_, fd);
}

int
gismoUtils::unregisterLocalMemory(uintptr_t addr, gismoRpcHandler *rpc_handler) {
    // remove this file firstly
    std::unique_lock<std::shared_mutex> guard(memoryMtx_);
    auto fpath = generateFileForMemory(localAgentId_, addr);
    deleteFileCache(fpath.c_str());
    auto ret = mvfs_unlink_mc(connection_, fpath.c_str());
    if (ret) {
        NIXL_WARN << "Failed to delete " << fpath << " in " << localAgentId_ << ", error " << ret;
    }
    rpc_handler->broadcastFileDelete(fpath.c_str());
    // clean the record in register file
    auto mem_file = std::string(ROOT_DIR) + localAgentId_ + "/" + MEM_REGISTER_INFO;
    auto delete_line = absl::StrFormat("%lu", addr);
    removeLineFromFile(mem_file.c_str(), delete_line.c_str());

    return ret;
}

int
gismoUtils::loadRemoteMemInfo(const std::string &remote_agent,
                              std::vector<nixlGismoBackendMD *> &backend_md_list,
                              bool force_reload) {
    auto fpath = std::string(ROOT_DIR) + remote_agent + "/" + MEM_REGISTER_INFO;
    mvfs_file_attr attr;
    if (force_reload) {
        auto it = remoteMemInfo_.get(remote_agent);
        assert(it);
        for (auto md : it.value().second) {
            delete md;
        }
        remoteMemInfo_.remove(remote_agent);
    }
    auto ret = getFileAttribute(fpath.c_str(), &attr);
    if (ret < 0) {
        NIXL_ERROR << "Failed to get file attribute for " << fpath << " in " << localAgentId_
                   << ", error " << ret;
        return ret;
    }
    auto it = remoteMemInfo_.get(remote_agent);
    if (!it) {
        // not loaded yet
        ret = pollMemoryRegisterInfo(remote_agent, backend_md_list);
        if (ret < 0) {
            NIXL_ERROR << "Failed to poll memory register info from " << remote_agent << " in "
                       << localAgentId_;
            return ret;
        }
        remoteMemInfo_.put(remote_agent, std::make_pair(attr.mtime, backend_md_list));
    } else {
        // already loaded before
        if (attr.mtime != it.value().first) {
            for (auto &md : it.value().second) {
                delete md;
            }
            backend_md_list.clear();
            // need to reload
            ret = pollMemoryRegisterInfo(remote_agent, backend_md_list);
            if (ret < 0) {
                NIXL_ERROR << "Failed to poll memory register info from " << remote_agent;
                return ret;
            }
            remoteMemInfo_.put(remote_agent, std::make_pair(attr.mtime, backend_md_list));
        } else {
            backend_md_list = it.value().second;
        }
    }
    return 0;
}

int
gismoUtils::pollMemoryRegisterInfo(const std::string &remote_agent,
                                   std::vector<nixlGismoBackendMD *> &backend_list) {
    auto fpath = std::string(ROOT_DIR) + remote_agent + "/" + MEM_REGISTER_INFO;
    const int buf_size = 1024 * 32;
    char *content = new char[buf_size];
    memset(content, 0, buf_size);
    auto ret = getSmallFile(fpath.c_str(), content, buf_size - 1);
    if (ret < 0) {
        NIXL_ERROR << "Failed to read " << fpath << " in " << localAgentId_ << ", error " << ret;
        return ret;
    };
    auto lines = str_split(content, "\n");
    for (auto &l : lines) {
        NIXL_DEBUG << "Read registered line " << l;
        nixlGismoBackendMD *md = new nixlGismoBackendMD(true);
        md->fromString(l);
        backend_list.push_back(md);
        NIXL_DEBUG << "Created md " << md->toString();
    }
    delete[] content;
    return 0;
}

nixl_status_t
gismoUtils::registerFileHandle(int fd, std::string &fp) {
    auto pos = lseek(fd, 0, SEEK_CUR);
    if (pos == (off_t)-1) {
        NIXL_ERROR << "Failed to get file pos, error " << errno;
        return NIXL_ERR_INVALID_PARAM;
    }
    auto current_time = getCurrentTime();
    fp = generateFileForMemory(localAgentId_, current_time); // create a random file for it
    mvfs_handle mv_fd;
    auto file_option = mvfs_file_default_option;
    file_option.chunk_size = configParams_.chunk_size_;
    file_option.flags |= MVFS_FILE_ALLOW_REWRITE;
    auto ret =
        mvfs_open_mc(connection_, &mv_fd, fp.c_str(), O_RDWR | O_CREAT, ACCESSPERMS, &file_option);
    if (ret) {
        NIXL_ERROR << "Failed to open file " << fp << " in " << localAgentId_ << ", error "
                   << errno;
        return (nixl_status_t)ret;
    }

    size_t read_offset = 0;
    ssize_t buf_size = configParams_.chunk_size_;
    nixl_status_t final_ret = NIXL_SUCCESS;
    auto read_buf = std::make_unique<char[]>(buf_size);
    do {
        auto bytes = pread(fd, read_buf.get(), buf_size, read_offset);
        if (bytes > 0) {
            auto written_bytes =
                mvfs_write_mc(connection_, mv_fd, read_buf.get(), bytes, read_offset);
            if (written_bytes != bytes) {
                NIXL_ERROR << "Failed to write " << fp << " in " << localAgentId_ << ", err "
                           << errno;
                final_ret = NIXL_ERR_INVALID_PARAM;
                break;
            }
            read_offset += bytes;
        }
        // reach end of the file
        if (bytes < buf_size) {
            break;
        }
    } while (1);
    lseek(fd, pos, SEEK_SET); // move back offset
    mvfs_close_mc(connection_, mv_fd);

    return final_ret;
}

int
gismoUtils::unregisterFileHandle(int fd, const std::string &fp) {
    mvfs_unlink_mc(connection_, fp.c_str());
    return 0;
}

int
gismoUtils::doFileRead(openFdMeta *fd_meta, bool gpu, uintptr_t dst_buf, int offset, size_t len)
    const {
    vram_operation op = gpu ? COPY_HOST_TO_DEV : COPY_HOST_TO_HOST;
    if (fd_meta->mapped_addr_ != 0) {
        // already mmaped
        mvfs_memcpy_vram((void *)dst_buf, (void *)(fd_meta->mapped_addr_ + offset), len, false, op);
        return len;
    }
    auto ret = mvfs_read_vram(connection_, fd_meta->fd_, (void *)dst_buf, len, offset, op);
    if (ret < 0) {
        NIXL_ERROR << "Failed to read " << uintptr_t(fd_meta->fd_) << " in " << localAgentId_
                   << ", error " << errno;
        return ret;
    }
    return ret;
}

int
gismoUtils::readFile(const std::string &src_file,
                     bool gpu,
                     uintptr_t dst_buf,
                     size_t len,
                     int offset) {
    auto fd_meta = getMvfsHandle(src_file.c_str(), true);
    if (fd_meta == nullptr) {
        NIXL_ERROR << "Failed to open file " << src_file << " in " << localAgentId_ << ", error "
                   << errno;
        return -errno;
    }
    return doFileRead(fd_meta, gpu, dst_buf, offset, len);
}

int
gismoUtils::readFileSegments(const std::string &dest_file,
                             bool gpu,
                             const uintptr_t start_addr,
                             const std::vector<memoryMeta> *segments) {
    if (segments->empty()) {
        return 0;
    }
    auto fd_meta = getMvfsHandle(dest_file.c_str(), true);
    if (fd_meta == nullptr) {
        NIXL_ERROR << "Failed to open file " << dest_file << " in " << localAgentId_ << ", error "
                   << errno;
        return -errno;
    }
    int ret = 0;
    if (segments->size() == 1 || !threadPool_->canCreateNewThread()) {
        for (auto &seg : *segments) {
            ret = doFileRead(fd_meta, gpu, seg.addr_, seg.addr_ - start_addr, seg.len_);
            if (ret < 0) {
                NIXL_ERROR << "Failed to read " << dest_file << " in " << localAgentId_ << ", ret "
                           << ret << ", errno " << errno;
                break;
            }
        }

    } else {
        auto sem = std::make_shared<gismoSemaphore>(segments->size());
        for (auto &seg : *segments) {
            threadPool_->enqueue([this, fd_meta, seg, start_addr, dest_file, sem, gpu](void) {
                auto read_ret =
                    doFileRead(fd_meta, gpu, seg.addr_, seg.addr_ - start_addr, seg.len_);
                if (read_ret < 0) {
                    NIXL_ERROR << "Failed to read " << dest_file << " in " << localAgentId_
                               << ", ret " << read_ret << ", errno " << errno;
                }
                sem->signal();
            });
        }
        // wait until all semaphore signaled
        sem->waitAll();
    }
    return ret;
}

int
gismoUtils::allocFile(const std::string &dest_file, size_t len) {
    mvfs_handle fd;
    auto file_option = mvfs_file_default_option;
    file_option.chunk_size = configParams_.chunk_size_;
    file_option.flags |= MVFS_FILE_ALLOW_REWRITE;
    auto ret = mvfs_open_mc(
        connection_, &fd, dest_file.c_str(), O_RDWR | O_CREAT, ACCESSPERMS, &file_option);
    if (ret) {
        NIXL_ERROR << "Failed to open file " << dest_file << " in " << localAgentId_ << ", error "
                   << errno;
        return ret;
    }
    ret = mvfs_fallocate_mc(connection_, fd, 0, 0, len);
    if (ret < 0) {
        NIXL_ERROR << "Failed to allocate space for " << dest_file << " in " << localAgentId_
                   << " with size " << len << ", ret " << ret << ", error " << errno;
        mvfs_close_mc(connection_, fd);
        return ret;
    }
    if (!configParams_.use_mmap_) {
        openFdMeta *fd_meta = new openFdMeta{fd, 0};
        openedMvfsFds_.put(dest_file, fd_meta);
        // preload the file into memory
        ret = mvfs_preload_mc(connection_, fd, 0, len);
        if (ret < 0) {
            NIXL_WARN << "Failed to preload " << dest_file << " in " << localAgentId_ << ", error "
                      << ret;
        }
        return 0;
    }

    auto mapped_addr = mvfs_prot_mmap_mc(connection_, fd, 0, len, PROT_READ | PROT_WRITE);
    if (mapped_addr == MAP_FAILED) {
        NIXL_ERROR << "Failed to mmap " << dest_file << " in " << localAgentId_ << " with size "
                   << len << ", error " << errno;
        mvfs_close_mc(connection_, fd);
        return -ENOMEM;
    }
    openFdMeta *fd_meta = new openFdMeta{fd, (uintptr_t)mapped_addr, len};
    openedMvfsFds_.put(dest_file, fd_meta);
    if (hasCudaDevices_) {
        cudaHostRegister(mapped_addr, len, cudaHostRegisterPortable);
    }
    return 0;
};

size_t
gismoUtils::doFileWrite(openFdMeta *fd_meta, bool gpu, uintptr_t src_data, int offset, size_t len)
    const {
    vram_operation cuda_op = gpu ? COPY_DEV_TO_HOST : COPY_HOST_TO_HOST;
    if (fd_meta->mapped_addr_ != 0) {
        // already mmaped
        mvfs_memcpy_vram(
            (void *)(fd_meta->mapped_addr_ + offset), (void *)src_data, len, true, cuda_op);
        return len;
    }
    const int max_batch_size = configParams_.chunk_size_;
    size_t written_len = 0;
    uint32_t batch_size = max_batch_size;
    if (len < batch_size) {
        batch_size = len;
    }
    // if gpu, then it means the incoming memory is in device
    auto batches = len / batch_size;
    if (len % batch_size) {
        batches++;
    }
    if (batches == 1) {
        // write all content directly in one call
        auto ret =
            mvfs_write_vram(connection_, fd_meta->fd_, (void *)src_data, len, offset, cuda_op);
        if (ret < 0) {
            NIXL_ERROR << "Failed to write file in " << localAgentId_ << ", ret " << ret
                       << ", error " << errno;
            return ret;
        }
        written_len = ret;
    } else if (!threadPool_->canCreateNewThread()) {
        // should be split but no thread can be created
        // need to write them one by one
        for (size_t i = 0; i < batches; ++i) {
            auto batch_len = batch_size;
            if (i == batches - 1) {
                batch_len = len - batch_size * i;
            }
            auto ret = mvfs_write_vram(connection_,
                                       fd_meta->fd_,
                                       (void *)(src_data + batch_size * i),
                                       batch_len,
                                       offset + batch_size * i,
                                       cuda_op);
            if (ret < 0) {
                NIXL_ERROR << "Failed to write file in " << localAgentId_ << ", ret " << ret
                           << ", error " << errno;
                break;
            }
            written_len += batch_len;
        }
    } else {
        // split to multiple batches and write in parallel
        auto sem = std::make_shared<gismoSemaphore>(batches);
        for (size_t i = 0; i < batches; ++i) {
            auto batch_len = batch_size;
            if (i == batches - 1) {
                batch_len = len - batch_size * i;
            }
            threadPool_->enqueue(
                [this, fd_meta, src_data, batch_size, batch_len, i, offset, sem, cuda_op]() {
                    auto ret = mvfs_write_vram(connection_,
                                               fd_meta->fd_,
                                               (void *)(src_data + batch_size * i),
                                               batch_len,
                                               offset + batch_size * i,
                                               cuda_op);
                    if (ret < 0) {
                        NIXL_ERROR << "Failed to write file in " << localAgentId_ << ", ret " << ret
                                   << ", error " << errno;
                    }
                    sem->signal();
                });
            written_len += batch_len;
        }
        sem->waitAll();
    }
    return written_len;
}

int
gismoUtils::writeFileSegments(const std::string &dst_file,
                              bool gpu,
                              const uintptr_t start_addr,
                              const std::vector<memoryMeta> *segments) {
    if (segments->empty()) {
        return 0;
    }
    auto fd_meta = getMvfsHandle(dst_file.c_str(), true);
    if (fd_meta == nullptr) {
        NIXL_ERROR << "Failed to open file " << dst_file << " in " << localAgentId_ << ", error "
                   << errno;
        return -errno;
    }
    int ret = 0;
    if (segments->size() == 1 || !threadPool_->canCreateNewThread()) {
        for (auto &seg : *segments) {
            ret = doFileWrite(fd_meta, gpu, seg.addr_, seg.addr_ - start_addr, seg.len_);
            if (ret < 0) {
                NIXL_ERROR << "Failed to write " << dst_file << " in " << localAgentId_
                           << ", error " << ret;
                break;
            }
        }
    } else {
        auto sem = std::make_shared<gismoSemaphore>(segments->size());
        for (auto &seg : *segments) {
            threadPool_->enqueue([this, fd_meta, gpu, seg, start_addr, dst_file, sem](void) {
                auto write_ret =
                    doFileWrite(fd_meta, gpu, seg.addr_, seg.addr_ - start_addr, seg.len_);
                if (write_ret < 0) {
                    NIXL_ERROR << "Failed to write " << dst_file << " in " << localAgentId_
                               << ", error " << write_ret;
                }
                sem->signal();
            });
        }
        // wait until all semaphore signaled
        sem->waitAll();
    }
    return ret;
}

int
gismoUtils::writeFile(const std::string &dst_file,
                      bool gpu,
                      uintptr_t src_data,
                      int offset,
                      size_t len) {
    // write file to DMO
    auto fd_meta = getMvfsHandle(dst_file.c_str(), true);
    if (fd_meta == nullptr) {
        NIXL_ERROR << "Failed to open file " << dst_file << " in " << localAgentId_ << ", error "
                   << errno;
        return -errno;
    }
    auto written_len = doFileWrite(fd_meta, gpu, src_data, offset, len);
    auto rc = errno;
    if (written_len != len) {
        NIXL_ERROR << "Failed to write file " << dst_file << " at " << offset << ", len " << len
                   << ", written " << written_len << " in " << localAgentId_ << ", err " << rc;
        return -rc;
    }
    NIXL_DEBUG << "Write file " << dst_file << ", fd " << (uintptr_t)fd_meta << " at " << offset
               << ", len " << len << ", at " << localAgentId_;
    return written_len;
}

std::string
gismoUtils::generateFileForMemory(const char *agent_id, uintptr_t addr) {
    return absl::StrFormat("%s%s/%lu", ROOT_DIR, agent_id, addr);
}

int
gismoUtils::putSmallFile(const char *fpath, const char *content, const size_t len) {
    mode_t mode = 0755;
    mvfs_file_option file_option;
    mvfs_init_file_option(&file_option);
    return mvfs_put_mc(connection_, fpath, mode, content, len, 0, &file_option);
}

int
gismoUtils::appendFile(const char *fpath, const char *content, const size_t len) {
    // get file attr for append
    mvfs_file_attr attr{
        .size = 0,
    };
    // ignore its error as we can create it directly
    mvfs_getattr_mc(connection_, fpath, &attr);

    mvfs_file_option file_option;
    mvfs_init_file_option(&file_option);
    file_option.flags |= MVFS_FILE_ALLOW_REWRITE;
    return mvfs_put_mc(connection_, fpath, 0755, content, len, attr.size, &file_option);
}

int
gismoUtils::getSmallFile(const char *fpath, char *content, int len) {
    return mvfs_get_mc(connection_, fpath, content, len, 0);
}

nixl_status_t
gismoUtils::offloadDesc(const nixl_xfer_op_t &operation,
                        const nixlMetaDesc &src_desc,
                        const nixlMetaDesc &dest_desc) {
    auto dst_backend = getBackendMDFromDesc(dest_desc);
    auto src_backend = getBackendMDFromDesc(src_desc);
    assert(dst_backend != nullptr && src_backend != nullptr);
    if (dst_backend->fileHandle_ == nullptr && src_backend->fileHandle_ == nullptr) {
        // for memory <-> memory
        cudaError_t copy_ret = cudaSuccess;
        if (!dst_backend->gpu_ && !src_backend->gpu_) {
            // dram to dram
            memcpy((void *)dest_desc.addr, (void *)src_desc.addr, src_desc.len);
        } else if (dst_backend->gpu_ && !src_backend->gpu_) {
            // dram to vram
            copy_ret = cudaMemcpy((void *)dest_desc.addr,
                                  (void *)src_desc.addr,
                                  src_desc.len,
                                  cudaMemcpyHostToDevice);
        } else if (dst_backend->gpu_ && src_backend->gpu_) {
            // vram to vram
            copy_ret = cudaMemcpy((void *)dest_desc.addr,
                                  (void *)src_desc.addr,
                                  src_desc.len,
                                  cudaMemcpyDeviceToDevice);
        } else if (!dst_backend->gpu_ && src_backend->gpu_) {
            // vram to dram
            copy_ret = cudaMemcpy((void *)dest_desc.addr,
                                  (void *)src_desc.addr,
                                  src_desc.len,
                                  cudaMemcpyDeviceToHost);
        }
        if (copy_ret != cudaSuccess) {
            NIXL_ERROR << "Failed to copy data from " << src_desc.addr << " to " << dest_desc.addr;
            return NIXL_ERR_BACKEND;
        }
    } else {
        // file <-> memory
        size_t op_bytes = 0;
        if (dst_backend->fileHandle_ == nullptr && src_backend->fileHandle_ != nullptr) {
            // read from file
            auto offset = dest_desc.addr - dst_backend->addr_;
            if (operation == NIXL_READ) {
                op_bytes = readFile(src_backend->fileHandle_->mount_point_,
                                    dst_backend->gpu_,
                                    dest_desc.addr,
                                    dest_desc.len,
                                    offset);
            } else {
                op_bytes = writeFile(src_backend->fileHandle_->mount_point_,
                                     dst_backend->gpu_,
                                     dest_desc.addr,
                                     offset,
                                     src_desc.len);
            }
        } else if (dst_backend->fileHandle_ != nullptr && src_backend->fileHandle_ == nullptr) {
            // read/write memory to file
            auto offset = src_desc.addr - src_backend->addr_;
            if (operation == NIXL_WRITE) {
                op_bytes = writeFile(dst_backend->fileHandle_->mount_point_,
                                     src_backend->gpu_,
                                     src_desc.addr,
                                     offset,
                                     src_desc.len);
            } else {
                op_bytes = readFile(dst_backend->fileHandle_->mount_point_,
                                    src_backend->gpu_,
                                    src_desc.addr,
                                    dest_desc.len,
                                    offset);
            }
        } else {
            // read/write one file to another file
            NIXL_ERROR << "Not support file to file copy";
            return NIXL_ERR_BACKEND;
        }
        if (op_bytes < 0) {
            NIXL_ERROR << "Failed to offload " << src_desc.len << ", error " << op_bytes;
            return NIXL_ERR_BACKEND;
        }
    }
    return NIXL_SUCCESS;
}

int
gismoUtils::offload(nixlGismoBackendReqH *handle) {
    assert(handle->local_.descCount() == handle->remote_.descCount());
    for (int i = 0; i < handle->local_.descCount(); ++i) {
        auto dst = handle->remote_[i];
        auto src = handle->local_[i];
        if (dst.len != src.len) {
            NIXL_ERROR << "#" << i << " segment size is not equal (src: " << src.len
                       << ", dst: " << dst.len << ")";
            return NIXL_ERR_INVALID_PARAM;
        }
        transferDesc(handle, src);
        threadPool_->enqueue([this, handle, src, dst](void) {
            auto offload_ret = offloadDesc(handle->operation_, src, dst);
            if (offload_ret != NIXL_SUCCESS) {
                NIXL_ERROR << "Failed to offload " << src.addr << " to " << dst.addr << ", error "
                           << offload_ret;
            }
            markDescDone(handle, src);
        });
    }
    return NIXL_IN_PROG;
}

void
gismoUtils::writeRemoteDesc(const nixlMetaDesc &dms,
                            const nixlMetaDesc &sms,
                            nixlGismoBackendReqH *handle,
                            gismoRpcHandler *rpc_handler) {
    struct timeval t_start, t_write_end, t_end;
    if (configParams_.record_metrics_) {
        gettimeofday(&t_start, nullptr);
    }
    auto remote_backend = getBackendMDFromDesc(dms);
    auto local_backend = getBackendMDFromDesc(sms);
    auto offset = dms.addr - remote_backend->addr_; // get offset of desc
    auto dst_path = generateFileForMemory(handle->remote_agent_.c_str(), remote_backend->addr_);
    auto ret = writeFile(dst_path, local_backend->gpu_, sms.addr, offset, sms.len);
    if (ret < 0) {
        NIXL_ERROR << "Failed to write " << dst_path << " in " << localAgentId_ << ", err " << ret;
        return;
    }
    NIXL_DEBUG << "Wrote " << dst_path << " in " << localAgentId_ << ", offset " << offset
               << ", len " << sms.len;
    if (configParams_.record_metrics_) {
        gettimeofday(&t_write_end, nullptr);
    }
    // update remote agent
    ret = rpc_handler->updateRemoteAgent(handle, dms, sms, remote_backend->addr_);
    if (ret < 0) {
        NIXL_ERROR << "Failed to update remote agent, error " << ret;
        return;
    }
    // first check whether request is completed
    if (isRequestCompleted(handle, &sms)) {
        notif_list_t notifs;
        notifs.push_back(std::make_pair(handle->remote_agent_, handle->opt_args_->notifMsg));
        NIXL_DEBUG << "append notify from " << handle->remote_agent_ << ", request "
                   << (uintptr_t)handle << " in " << localAgentId_;
        rpc_handler->appendNotifs(notifs);
    }
    // then mark og-going desc completed
    markDescDone(handle, sms);

    if (configParams_.record_metrics_) {
        gettimeofday(&t_end, nullptr);
        perfData_.update(dms.len, [t_start, t_write_end, t_end](metricsItem &item) {
            item.total_count_++;
            item.total_time_ +=
                (((t_end.tv_sec - t_start.tv_sec) * 1e6) + (t_end.tv_usec - t_start.tv_usec));
            item.total_write_time_ += (((t_write_end.tv_sec - t_start.tv_sec) * 1e6) +
                                       (t_write_end.tv_usec - t_start.tv_usec));
        });
    }
}

void
gismoUtils::readRemoteDesc(const nixlMetaDesc &dms,
                           const nixlMetaDesc &sms,
                           nixlGismoBackendReqH *handle,
                           gismoRpcHandler *rpc_handler) {
    auto remote_backend = dynamic_cast<nixlGismoBackendMD *>(dms.metadataP);
    auto local_backend = dynamic_cast<nixlGismoBackendMD *>(sms.metadataP);
    auto dst_path = generateFileForMemory(handle->remote_agent_.c_str(), remote_backend->addr_);
    auto offset = dms.addr - remote_backend->addr_;
    // use local backend memory type
    auto ret = readFile(dst_path, local_backend->gpu_, sms.addr, sms.len, offset);
    if (ret < 0) {
        NIXL_ERROR << "Failed to read " << dst_path << ", ret " << ret << ", err " << errno;
        return;
    }
    // update remote agent
    ret = rpc_handler->updateRemoteAgent(handle, dms, sms, remote_backend->addr_);
    if (ret < 0) {
        NIXL_ERROR << "Failed to update remote agent, error " << ret;
        return;
    }
    markDescDone(handle, sms);
    NIXL_DEBUG << "Marked request " << (uintptr_t)handle << " desc " << (uintptr_t)sms.addr
               << " done, is request done: " << isRequestCompleted(handle, &sms);
}

int
gismoUtils::transfer(nixlGismoBackendReqH *handle, gismoRpcHandler *rpc_handler) {
    assert(handle->local_.descCount() == handle->remote_.descCount());
    if (handle->operation_ == NIXL_WRITE) {
        // write to DMO
        for (int i = 0; i < handle->local_.descCount(); ++i) {
            auto dms = handle->remote_[i];
            auto sms = handle->local_[i];
            transferDesc(handle, sms);
            threadPool_->enqueue([dms, sms, this, handle, rpc_handler](void) {
                writeRemoteDesc(dms, sms, handle, rpc_handler);
            });
        }
        return NIXL_IN_PROG;
    } else if (handle->operation_ == NIXL_READ) {
        // read from DMO
        std::unordered_map<uintptr_t, bool> submitted_map;
        for (; submitted_map.size() < size_t(handle->local_.descCount());) {
            for (int i = 0; i < handle->local_.descCount(); ++i) {
                auto dms = handle->remote_[i];
                auto sms = handle->local_[i];
                // record submitted local desc as remote desc may be duplicated
                if (submitted_map.count(sms.addr) > 0) {
                    continue;
                }
                // check whether remote desc is updated
                if (reqMgr_->getDescUpdateTime((uintptr_t)handle, dms.addr) == 0) {
                    continue;
                }
                transferDesc(handle, sms);
                submitted_map[sms.addr] = true;
                threadPool_->enqueue([dms, sms, this, handle, rpc_handler](void) {
                    readRemoteDesc(dms, sms, handle, rpc_handler);
                });
            }
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
        return NIXL_IN_PROG;
    }
    return 0;
};

int
gismoUtils::getFileAttribute(const char *fpath, mvfs_file_attr *attr) {
    auto ret = mvfs_getattr_mc(connection_, fpath, attr);
    if (ret != 0) {
        return ret;
    }
    return 0;
}

bool
gismoUtils::fileExists(const char *fpath) {
    mvfs_file_attr attr;
    return getFileAttribute(fpath, &attr) == 0;
}

int
gismoUtils::mkdir(const char *fp) {
    return mvfs_mkdir_mc(connection_, fp, ACCESSPERMS);
}

int
gismoUtils::rmdir(const char *fp) {
    return mvfs_rmdir_mc(connection_, fp);
}

int
gismoUtils::rmnode(const char *fp) {
    return mvfs_unlink_mc(connection_, fp);
}

int
gismoUtils::mknode(const char *fp) {
    mvfs_file_option file_option;
    mvfs_init_file_option(&file_option);
    return mvfs_mknod_mc(connection_, fp, ACCESSPERMS, &file_option);
}

openFdMeta *
gismoUtils::getMvfsHandle(const char *fp, bool preload) {
    auto ret = openedMvfsFds_.getAndPut(fp, [this, preload](std::string &sp) {
        openFdMeta *fd_meta = nullptr;
        mvfs_handle fd = nullptr;
        auto file_option = mvfs_file_default_option;
        file_option.chunk_size = configParams_.chunk_size_;
        file_option.flags |= MVFS_FILE_ALLOW_REWRITE;
        auto rc = mvfs_open_mc(connection_, &fd, sp.c_str(), O_RDWR, ACCESSPERMS, &file_option);
        if (rc != 0) {
            NIXL_ERROR << "Failed to open file " << sp << " in " << localAgentId_ << ", ret " << rc
                       << ", error " << errno;
            return fd_meta;
        }
        mvfs_file_attr attr;
        rc = mvfs_fgetattr_mc(connection_, fd, &attr);
        if (rc != 0) {
            NIXL_ERROR << "Failed to get attribute for " << sp << " in " << localAgentId_
                       << ", error " << rc;
            mvfs_close_mc(connection_, fd);
            return fd_meta;
        }
        if (configParams_.use_mmap_) {
            auto mapped_addr =
                mvfs_prot_mmap_mc(connection_, fd, 0, attr.size, PROT_READ | PROT_WRITE);
            if (mapped_addr == MAP_FAILED) {
                NIXL_ERROR << "Failed to mmap " << sp << " with size " << attr.size << " in "
                           << localAgentId_ << ", error " << errno;
                mvfs_close_mc(connection_, fd);
                return fd_meta;
            }
            if (hasCudaDevices_) {
                cudaHostRegister(mapped_addr, attr.size, cudaHostRegisterPortable);
            }
            fd_meta = new openFdMeta{fd, (uintptr_t)mapped_addr, attr.size};
        } else {
            if (preload) {
                // preload the file into cache
                rc = mvfs_preload_mc(connection_, fd, 0, attr.size);
                if (rc < 0) {
                    NIXL_WARN << "Failed to preload " << sp << " in " << localAgentId_ << ", error "
                              << rc;
                }
            }
            fd_meta = new openFdMeta{fd, 0};
        }
        return fd_meta;
    });

    return ret;
}

int
gismoUtils::listAllAgents(std::vector<std::string> &agents) {
    char **list = nullptr;
    uint64_t obj_number = 0;
    uint64_t session_id = 0;

    auto ret = mvfs_list_mc(connection_, ROOT_DIR, &list, &obj_number, &session_id);
    if (ret != 0) {
        NIXL_ERROR << "Failed to list " << ROOT_DIR << ", error " << errno;
        return ret;
    }
    for (uint64_t index = 0; index < obj_number; index++) {
        std::string name = list[index];
        agents.push_back(name.substr(0, name.size() - 1));
    }

    mvfs_free_list_mc(connection_, list, obj_number);
    return 0;
}

void
gismoUtils::deleteFileCache(const char *fp) {
    auto fd_meta = openedMvfsFds_.get(fp);
    if (fd_meta) {
        NIXL_DEBUG << "Ready to delete " << fp << " with fd " << (uintptr_t)fd_meta.value()->fd_
                   << " in " << localAgentId_;
        if (fd_meta.value()->mapped_addr_ != 0) {
            if (hasCudaDevices_) {
                cudaHostUnregister((void *)fd_meta.value()->mapped_addr_);
            }
            mvfs_munmap_mc(
                connection_, (void *)fd_meta.value()->mapped_addr_, fd_meta.value()->mapped_len_);
        }
        mvfs_close_mc(connection_, fd_meta.value()->fd_);
        delete fd_meta.value();
    }
    openedMvfsFds_.remove(fp);
}

bool
gismoUtils::isRequestCompleted(nixlGismoBackendReqH *req, const nixlMetaDesc *ongoing_desc) {
    for (int i = 0; i < req->local_.descCount(); ++i) {
        auto desc = req->local_[i];
        // skip ongoing desc check
        if (ongoing_desc != nullptr && desc.addr == ongoing_desc->addr) {
            continue;
        }
        if (!isDescCompleted(req, desc)) {
            return false;
        }
    }
    return true;
}

uintptr_t
strToInt(const std::string &str) {
    return std::stoul(str);
}

uint64_t
getCurrentTime() {
    auto now = std::chrono::high_resolution_clock::now();
    auto nanoseconds_since_epoch =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());
    return nanoseconds_since_epoch.count();
}