#!/bin/bash
# This script integrates Gismo into the system by setting up necessary environment variables
# and updating the PATH. It also verifies the installation.

export DMO_SOCKET_PATH=${DMO_SOCKET_PATH:-/tmp/dmo.daemon.sock.sheperd}
ETCD_PATH=${ETCD_PATH:---etcd_endpoints http://localhost:2379}
GISMO_DIR=${GISMO_DIR:-"/opt/memverge/MemVergeDMO"}
NIXL_DIR=${NIXL_DIR:-"/opt/nvidia/nvda_nixl"}

BUILD_NIXL=${BUILD_NIXL:-1}

function setup_gismo() {
  # Check if Gismo directory exists
  if [ ! -d "$GISMO_DIR" ]; then
      echo "Gismo directory not found at $GISMO_DIR. Please install Gismo first."
      return 1
  fi

  # Set environment variables
  export GISMO_HOME="${GISMO_DIR}"
  export PATH="${GISMO_HOME}/bin:${PATH}"

  # Verify installation
  if command $GISMO_DIR/bin/dmocli ls / --socket_path $DMO_SOCKET_PATH >/dev/null 2>&1; then
    echo "Gismo has been successfully integrated into the system."
    echo "GISMO_HOME is set to ${GISMO_HOME}"
    echo "Gismo version: $(dmo_daemon -v)"
  else
    echo "Failed to integrate Gismo. Please check the installation at ${GISMO_HOME}."
    return 1
  fi
}

function build_nixl() {
    echo "Building NixL from source..."
    # Navigate to NixL source directory
    NIXL_SRC_DIR=$(realpath .)
    cd $NIXL_SRC_DIR || { echo "NixL source directory not found."; return 1; }

    if [ -d "build" ]; then
        meson setup build --reconfigure
    else
        meson setup build
    fi
    # Run build commands (example using make)
    cd build
    ninja
    sudo ninja install

    if [ $? -eq 0 ]; then
        echo "NixL built successfully."
    else
        echo "Failed to build NixL."
        return 1
    fi

    cd $NIXL_SRC_DIR/benchmark/nixlbench
    meson setup -Dprefix=$NIXL_DIR build
    cd build
    ninja
    sudo ninja install
    cd $NIXL_SRC_DIR
}

function run_tests() {
    echo "Running Gismo and NixL integration tests..."
    cleanup
    # actual test commands
    $NIXL_DIR/bin/nixl_gismo_test
    if [ $? -eq 0 ]; then
         echo "All tests passed successfully."
    else
         echo "Some tests failed. Please check the logs for details."
         return 1
    fi
    cleanup
    ulimit -n 65535
    $NIXL_DIR/bin/nixl_gismo_file_test
}

function run_benchmark() {
    echo "Running Gismo and NixL benchmark tests..."
    cleanup
    THREAD_NUM=8
    # actual test commands
    $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --Gismo_offload_mode --initiator_seg_type DRAM --op_type WRITE --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH --filepath /tmp
    $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --Gismo_offload_mode --initiator_seg_type DRAM --op_type READ --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH --filepath /tmp
    $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --Gismo_offload_mode --Gismo_use_mmap --initiator_seg_type DRAM --op_type WRITE --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH --filepath /tmp
    $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --Gismo_offload_mode --Gismo_use_mmap --initiator_seg_type DRAM --op_type READ --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH --filepath /tmp
    if has_gpu; then
        cleanup
        $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --Gismo_offload_mode --initiator_seg_type VRAM --op_type WRITE --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH --filepath /tmp
        $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --Gismo_offload_mode --initiator_seg_type VRAM --op_type READ --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH --filepath /tmp
        $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --Gismo_offload_mode --Gismo_use_mmap --initiator_seg_type VRAM --op_type WRITE --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH --filepath /tmp
        $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --Gismo_offload_mode --Gismo_use_mmap --initiator_seg_type VRAM --op_type READ --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH --filepath /tmp
    else
        echo "GPU not detected, skipping VRAM offload benchmark."
    fi
    
    cleanup
    $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --initiator_seg_type DRAM --op_type READ --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH &
    $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --initiator_seg_type DRAM --op_type READ --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH

    cleanup
    $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --Gismo_use_mmap --initiator_seg_type DRAM --op_type READ --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH &
    $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --Gismo_use_mmap --initiator_seg_type DRAM --op_type READ --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH

    cleanup
    $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --initiator_seg_type DRAM --op_type WRITE --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH &
    $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --initiator_seg_type DRAM --op_type WRITE --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH

    cleanup
    $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --Gismo_use_mmap --initiator_seg_type DRAM --op_type WRITE --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH &
    $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --Gismo_use_mmap --initiator_seg_type DRAM --op_type WRITE --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH
    if has_gpu; then
        cleanup
        $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --initiator_seg_type VRAM --op_type READ --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH &
        $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --initiator_seg_type VRAM --op_type READ --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH
        cleanup
        $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --Gismo_use_mmap --initiator_seg_type VRAM --op_type READ --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH &
        $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --Gismo_use_mmap --initiator_seg_type VRAM --op_type READ --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH
        cleanup
        $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --initiator_seg_type VRAM  --op_type WRITE --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH &
        $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --initiator_seg_type VRAM  --op_type WRITE --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH
        cleanup
        $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --Gismo_use_mmap --initiator_seg_type VRAM  --op_type WRITE --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH &
        $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --Gismo_use_mmap --initiator_seg_type VRAM  --op_type WRITE --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH

        cleanup
        $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo  --initiator_seg_type VRAM --target_seg_type VRAM  --op_type WRITE --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH &
        $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo  --initiator_seg_type VRAM --target_seg_type VRAM  --op_type WRITE --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH

        cleanup
        $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --Gismo_use_mmap --initiator_seg_type VRAM --target_seg_type VRAM  --op_type WRITE --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH &
        $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --Gismo_use_mmap --initiator_seg_type VRAM --target_seg_type VRAM  --op_type WRITE --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH

        cleanup
        $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo  --initiator_seg_type VRAM --target_seg_type VRAM  --op_type READ --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH &
        $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo  --initiator_seg_type VRAM --target_seg_type VRAM  --op_type READ --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH

        cleanup
        $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --Gismo_use_mmap --initiator_seg_type VRAM --target_seg_type VRAM  --op_type READ --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH &
        $NIXL_DIR/bin/nixlbench $ETCD_PATH --backend Gismo --Gismo_use_mmap --initiator_seg_type VRAM --target_seg_type VRAM  --op_type READ --num_threads $THREAD_NUM --Gismo_sock_path $DMO_SOCKET_PATH
        
    else
        echo "GPU not detected, skipping VRAM transfer benchmark."
    fi
}

function cleanup() {
    echo "Cleaned up data in dmo."
    #dmocli rmdir / --socket_path $DMO_SOCKET_PATH
    dmocli df -h --socket_path $DMO_SOCKET_PATH
}

function has_gpu() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

function main() {
    setup_gismo
    if [ $? -ne 0 ]; then
        exit 1
    fi

    if [ "$BUILD_NIXL" -eq 1 ]; then
        build_nixl
        if [ $? -ne 0 ]; then
            exit 1
        fi
    fi

    echo "Gismo and NixL integration completed successfully. Ready to run tests and benchmarks."
    
    if has_gpu; then
        echo "GPU detected in the system, turn on GPU support."
        sudo ${GISMO_DIR}/bin/enable-gpu on
    fi
    export LD_LIBRARY_PATH=$GISMO_HOME/lib:$LD_LIBRARY_PATH:$NIXL_DIR/lib/x86_64-linux-gnu:$NIXL_DIR/lib/x86_64-linux-gnu/plugins
    echo "Running NIXL tests..."
    run_tests

    echo "Resetting etcd data..."
    sudo systemctl stop etcd || { echo "etcd not found"; exit 1; }
    sudo rm -rf /var/lib/etcd/*
    sudo systemctl start etcd

    echo "Running NIXL benchmarks..."
    run_benchmark
}

main
