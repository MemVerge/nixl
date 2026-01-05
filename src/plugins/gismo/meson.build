# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Use the gismo_lib_path from parent meson.build
if fs.is_absolute(dmo_path)
  gismo_inc_path = join_paths(dmo_path, 'include')
else
  gismo_inc_path = join_paths('../../..', dmo_path, 'include')
endif
gismo_dep = declare_dependency(
  link_args : ['-L' + gismo_lib_path, '-l' + gismo_lib_file],
  include_directories : include_directories(gismo_inc_path))

if 'Gismo' in static_plugins
  gismo_backend_lib = static_library('Gismo',
                    'gismo_utils.cpp', 'gismo_utils.h','gismo_thread_pool.h',
                    'gismo_backend.cpp', 'gismo_backend.h',
                    'gismo_plugin.cpp','gismo_rpc_handler.cpp',
                    dependencies: [nixl_infra, gismo_dep, cuda_dep, nixl_common_dep, file_utils_interface],
                    include_directories: [nixl_inc_dirs, utils_inc_dirs],
                    install: false,
                    cpp_args : compile_flags,
                    name_prefix: 'libplugin_')  # Custom prefix for plugin libraries
else
  gismo_backend_lib = shared_library('Gismo',
                    'gismo_utils.cpp', 'gismo_utils.h','gismo_thread_pool.h',
                    'gismo_backend.cpp', 'gismo_backend.h',
                    'gismo_plugin.cpp','gismo_rpc_handler.cpp',
                    dependencies: [nixl_infra, gismo_dep, cuda_dep, nixl_common_dep, file_utils_interface],
                    include_directories: [nixl_inc_dirs, utils_inc_dirs],
                    install: true,
                    cpp_args : ['-fPIC','-rdynamic'],
                    name_prefix: 'libplugin_',  # Custom prefix for plugin libraries
                    install_dir: plugin_install_dir,
                    install_rpath: '$ORIGIN/..')
  if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
                    'echo "gismo=' + gismo_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
                    check: true
                )
    endif
endif

gismo_backend_interface = declare_dependency(link_with: gismo_backend_lib)
