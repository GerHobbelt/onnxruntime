// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Standard libs/headers.
#include <limits>
#include <unordered_map>

// 1st-party libs/headers.
#include "core/framework/ortdevice.h"
#include "core/framework/provider_options.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/graph/constants.h"
#include "./amd_unified_execution_provider_utils.h"


namespace onnxruntime {

// User-defined information needed to construct an EP.
struct AMDUnifiedExecutionProviderInfo {
  // EP info should be a combination of all covered downstream EPs' info.
  // The key of this map can be either `ProviderOptions` or EP info.
  std::unordered_map<std::string, ProviderOptions> downstream_ep_infos;
  std::vector<std::string> device_types;

  explicit AMDUnifiedExecutionProviderInfo(const ProviderOptions&);

  explicit AMDUnifiedExecutionProviderInfo(const std::string&);

  AMDUnifiedExecutionProviderInfo() {
    AMDUnifiedExecutionProviderInfo("CPU");
  }

  const char* get_json_config_str() const {
    return json_config_.c_str();
  }

 private:
  ProviderOptions provider_options_;
  const std::string json_config_;
};

}  // namespace onnxruntime