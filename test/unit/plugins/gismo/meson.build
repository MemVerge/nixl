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

if fs.is_absolute(dmo_path)
  gismo_inc_path = join_paths(dmo_path, 'include')
else
  gismo_inc_path = join_paths('../../../..', dmo_path, 'include')
endif
gismo_backend_dep = declare_dependency(
  link_with: gismo_backend_lib,
  include_directories: [nixl_inc_dirs, '../../../../src/plugins/gismo',gismo_inc_path])

nixl_gismo_app = executable('nixl_gismo_test', 'nixl_gismo_test.cpp',
                          dependencies: [nixl_dep, nixl_infra,gismo_backend_dep,gismo_dep, cuda_dep, absl_log_dep],
                          include_directories: [nixl_inc_dirs, utils_inc_dirs],
                          install: true)

nixl_gismo_file = executable('nixl_gismo_file_test', 'nixl_gismo_file_test.cpp',
                          dependencies: [nixl_dep, nixl_infra,gismo_backend_dep,gismo_dep, cuda_dep, absl_log_dep],
                          include_directories: [nixl_inc_dirs, utils_inc_dirs],
                          install: true)
