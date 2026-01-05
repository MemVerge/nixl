# NIXL Gismo Plugin

This plugin utilizes MemVerge `libmvfs.so` as an I/O backend for NIXL.

## Usage
1. Install [libmvfs]
2. Build NIXL.
3. Once the Gismo Backend is built, you can use it in your data transfer task by specifying the backend name as "Gismo":

```cpp
nixl_status_t ret1;
std::string ret_s1;
nixlAgentConfig cfg(true);
nixl_b_params_t init1;
nixl_mem_list_t mems1;
nixlBackendH      *gismo;
nixlAgent A1(agent1, cfg);
init1["socket_path"]="dmo.daemon.sock.0"; // make sure it's aligned with DMO setup
ret1 = A1.getPluginParams("Gismo", mems1, init1);
assert (ret1 == NIXL_SUCCESS);
ret1 = A1.createBackend("Gismo", init1, gismo);
...
```

### Backend parameters
Paramaters accepted by the Gismo plugin during createBackend()
- sock_path: DMO sock path	

## Performance tuning
