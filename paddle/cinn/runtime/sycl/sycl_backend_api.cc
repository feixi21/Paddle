// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/cinn/runtime/sycl/sycl_backend_api.h"
#include <glog/logging.h>

namespace cinn {
namespace runtime {
namespace Sycl {
SYCLBackendAPI* SYCLBackendAPI::Global() {
  static auto* inst = new SYCLBackendAPI();
  return inst;
}

Arch SYCLBackendAPI::Init(Arch arch) {
  if (initialized_) return this->arch;
  // Target::Arch -> sycl::backend
  sycl::backend backend;
  arch.Match(
      [&](common::UnknownArch){
          SYCL_CALL(backend =
                   sycl::device::get_devices(sycl::info::device_type::gpu)[0]
                       .get_backend()); 
      },
      [&](common::X86Arch) { 
          SYCL_CALL(backend =
                    sycl::device::get_devices(sycl::info::device_type::gpu)[0]
                        .get_backend());
      },
      [&](common::ARMArch){ 
          SYCL_CALL(backend =
                    sycl::device::get_devices(sycl::info::device_type::gpu)[0]
                        .get_backend());
      },
      [&](common::NVGPUArch) {
          backend = sycl::backend::ext_oneapi_cuda;
      },
      [&](common::HygonDCUArchHIP){
        backend = sycl::backend::ext_oneapi_hip;
      },
      [&](common::HygonDCUArchSYCL){
        backend = sycl::backend::ext_oneapi_level_zero;
      });
  // look for matched devices
  for (auto device : sycl::device::get_devices(sycl::info::device_type::gpu)) {
    if (device.get_backend() == backend) {
      this->devices.push_back(device);
    }
  }
  if (this->devices.size() == 0) {
    LOG(FATAL) << "No valid gpu device matched given arch:";
  }
  this->contexts.resize(this->devices.size(), nullptr);
  this->queues.resize(this->devices.size());
  // sycl::backend -> Target::Arch
  switch (backend) {
    case sycl::backend::ext_oneapi_cuda:
      this->arch = cinn::common::NVGPUArch{};
      break;
    case sycl::backend::ext_oneapi_hip:
      this->arch = cinn::common::HygonDCUArchHIP{};
      break;
    case sycl::backend::ext_oneapi_level_zero:
      this->arch = cinn::common::HygonDCUArchSYCL{};
      break;
    default:
      LOG(FATAL) << "SYCL Not supported arch:";
  }
  initialized_ = true;
  set_device(0);
  return this->arch;
}

void SYCLBackendAPI::set_device(int device_id) {
  if (!initialized_) Init(cinn::common::UnknownArch {});
  if (device_id < 0) {
    LOG(FATAL) << "set valid device id! device id:" << device_id;
  } else if (device_id > this->devices.size() - 1) {
    LOG(FATAL) << "set valid device id! device id:" << device_id
               << " > max device id:" << this->devices.size() - 1;
  }
  if (this->contexts[device_id] == nullptr) {
    auto exception_handler = [](sycl::exception_list exceptions) {
      for (const std::exception_ptr& e : exceptions) {
        try {
          std::rethrow_exception(e);
        } catch (const sycl::exception& e) {
          LOG(INFO) << "Caught asynchronous SYCL exception:\n" << e.what();
        }
      }
    };
    sycl::property_list q_prop{
        sycl::property::queue::in_order()};  // In order queue
    // create context and queue
    this->contexts[device_id] =
        new sycl::context(this->devices[device_id], exception_handler);
    // one device one queue
    this->queues[device_id].push_back(new sycl::queue(
        *this->contexts[device_id], this->devices[device_id], q_prop));
  }
  this->now_device_id = device_id;
}

int SYCLBackendAPI::get_device() { return this->now_device_id; }

int SYCLBackendAPI::get_device_property(
    DeviceProperty device_property, std::optional<int> device_id) {
  int index = device_id.value_or(this->now_device_id);
  int rv = -1;

  switch (device_property) {
    case DeviceProperty::MaxBlockDimX: {
      rv = 101;
      break;
    }
    case DeviceProperty::MaxBlockDimY: {
      rv = 101;
      break;
    }
    case DeviceProperty::MaxBlockDimZ: {
      rv = 101;
      break;
    }
    case DeviceProperty::MaxGridDimX: {
      rv = 101;
      break;
    }
    case DeviceProperty::MaxGridDimY: {
      rv = 101;
      break;
    }
    case DeviceProperty::MaxGridDimZ: {
      rv = 101;
      break;
    }
    case DeviceProperty::MaxSharedMemoryPerBlock: {
      rv = this->devices[index].get_info<::sycl::info::device::local_mem_size>();
      break;
    }
    case DeviceProperty::MaxThreadsPerBlock: {
      rv = this->devices[index]
               .get_info<::sycl::info::device::max_work_group_size>();
      break;
    }
    case DeviceProperty::MaxThreadsPerSM: {
      rv = this->devices[index]
               .get_info<::sycl::info::device::max_work_group_size>();
      break;
    }
    case DeviceProperty::MultiProcessorCount: {
      rv = this->devices[index]
               .get_info<::sycl::info::device::max_compute_units>();
      break;
    }
    case DeviceProperty::MaxBlocksPerSM: {
      LOG(FATAL) << "SYCL Not supported device property : MaxBlocksPerSM !";
      break;
    }
    case DeviceProperty::WarpSize: {
      std::vector<size_t> sub_group_sizes =
          this->devices[index].get_info<::sycl::info::device::sub_group_sizes>();
      size_t max_sub_group_size =
          *max_element(std::begin(sub_group_sizes), std::end(sub_group_sizes));
      rv = static_cast<int>(max_sub_group_size);
    }
    default:
      // PADDLE_THROW(
      //     ::common::errors::InvalidArgument("Not supported device property!"));
      std::cout << "error!" << std::endl;
  }
  return rv;
}

void* SYCLBackendAPI::malloc(size_t numBytes) {
  if (now_device_id == -1) set_device(0);
  VLOG(3) << "sycl malloc";
  void* dev_mem = nullptr;
  SYCL_CALL(dev_mem = sycl::malloc_device(numBytes,
                                          this->devices[now_device_id],
                                          *this->contexts[now_device_id]));
  if (dev_mem == nullptr)
    LOG(ERROR) << "allocate sycl device memory failure!" << std::endl;
  return dev_mem;
}

void SYCLBackendAPI::free(void* data) {
  VLOG(3) << "sycl free";
  SYCL_CALL(sycl::free(data, *this->contexts[now_device_id]));
}

void SYCLBackendAPI::memset(void* data, int value, size_t numBytes) {
  VLOG(3) << "sycl memset";
  SYCL_CALL(
      this->queues[now_device_id][0]->memset(data, value, numBytes).wait());
}

void SYCLBackendAPI::memcpy(void* dest,
                            const void* src,
                            size_t numBytes,
                            MemcpyType type) {
  VLOG(3) << "sycl memcpy";
  sycl::queue* Q;
  switch (type) {
    case MemcpyType::HostToHost:
      Q = this->queues[now_device_id][0];
      break;
    case MemcpyType::HostToDevice:
      Q = this->queues[now_device_id][0];
      break;
    case MemcpyType::DeviceToHost:
      Q = this->queues[now_device_id][0];
      break;
    case MemcpyType::DeviceToDevice:
      Q = this->queues[now_device_id][0];
      break;
  }
  SYCL_CALL(Q->memcpy(dest, src, numBytes).wait());
}

void SYCLBackendAPI::device_sync() {
  VLOG(3) << "sycl device sync";
  if (now_device_id == -1) set_device(0);
  for (auto queues_in_one_device : this->queues) {
    for (auto queue : queues_in_one_device) {
      // LOG(INFO) << "sycl stream sync";
      SYCL_CALL(queue->wait_and_throw());
    }
  }
}

void SYCLBackendAPI::stream_sync(void* stream) {
  VLOG(3) << "sycl stream sync";
  SYCL_CALL(static_cast<sycl::queue*>(stream)->wait_and_throw());
}

sycl::queue* SYCLBackendAPI::get_now_queue() {
  return this->queues[now_device_id][0];
}

std::string SYCLBackendAPI::GetGpuVersion() {
  if (now_device_id == -1) set_device(0);
  sycl::device device = this->devices[now_device_id];
  sycl::backend backend = device.get_backend();
  switch (backend) {
    case sycl::backend::ext_oneapi_cuda: {
      std::string gpu_version = "sm_";
      std::string version_with_point =
          device.get_info<sycl::info::device::backend_version>();
      size_t pos = version_with_point.find(".");
      if (pos != std::string::npos) {
        gpu_version +=
            version_with_point.substr(0, pos) +
            version_with_point.substr(pos + 1, version_with_point.size());
      }
      return gpu_version;
    }
    case sycl::backend::ext_oneapi_hip: {
      std::string gpu_version = device.get_info<sycl::info::device::version>();
      size_t pos = gpu_version.find(":");
      if (pos != std::string::npos) gpu_version = gpu_version.substr(0, pos);
      return gpu_version;
    }
    case sycl::backend::ext_oneapi_level_zero:
      return "";
    default:
      LOG(ERROR) << "unknown sycl backend!";
  }
}

std::array<int, 3> SYCLBackendAPI::get_max_grid_dims(
    std::optional<int> device_id) {
    std::array<int, 3> kMaxGridDims;
    int index = device_id.value_or(this->now_device_id);
    kMaxGridDims = std::array<int, 3>{2097151, 2097151, 2097151};
    // ::sycl::id<3> max_work_item_sizes =
    //       this->devices[index]
    //           .get_info<::sycl::_V1::info::device::max_work_item_sizes>();
    //   kMaxGridDims = std::array<int, 3>{max_work_item_sizes[2],
    //                           max_work_item_sizes[1],
    //                           max_work_item_sizes[0]};
  return kMaxGridDims;
}

std::array<int, 3> SYCLBackendAPI::get_max_block_dims(
    std::optional<int> device_id) {
  std::array<int, 3> kMaxBlockDims;
  kMaxBlockDims = std::array<int, 3>{2097151, 2097151, 2097151};
  return kMaxBlockDims;
}

}  // namespace Sycl
}  // namespace runtime
}  // namespace cinn

// case DeviceProperty::MaxBlockDimX: {
      // rv = this->devices[index]
      //        .get_info<sycl::info::device::max_work_item_dimensions>();
      // std::array<int, 3> max_work_item_sizes;
      // max_work_item_sizes = this->devices[index].get_info<sycl::info::device::max_work_item_sizes>();
      //rv = max_sizes[0];
      // break;
    //}
    // case DeviceProperty::MaxBlockDimY: {
    //   ::sycl::id<3> max_work_item_sizes =
    //       this->devices[index]
    //           .get_info<::sycl::info::device::max_work_item_sizes>();
    //   rv = max_work_item_sizes[1];
    //   break;
    // }
    // case DeviceProperty::MaxBlockDimZ: {
    //   ::sycl::id<3> max_work_item_sizes =
    //       this->devices[index]
    //           .get_info<::sycl::info::device::max_work_item_sizes>();
    //   rv = max_work_item_sizes[2];
    //   break;
    // }
    // case DeviceProperty::MaxGridDimX: {
    //   ::sycl::id<3> grid_dims =
    //       this->devices[index].get_info<sycl::info::device::max_work_item_sizes<3>>();
    //   rv = grid_dim3[0];
    //   break;
    // }
    // case DeviceProperty::MaxGridDimY: {
    //   ::sycl::id<3> grid_dims =
    //       this->devices[index].get_info<sycl::info::device::max_work_item_sizes<3>>();
    //   rv = grid_dim3[1];
    //   break;
    // }
    // case DeviceProperty::MaxGridDimZ: {
    //   ::sycl::id<3> grid_dims =
    //       this->devices[index].get_info<sycl::info::device::max_work_item_sizes<3>>();
    //   rv = grid_dim3[2];
    //   break;
    // }