// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include "core/providers/tidl/tidl_provider_factory.h"
#include "tidl_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {

struct TidlProviderFactory : IExecutionProviderFactory {
  TidlProviderFactory(const std::string& type, const TIDLProviderOptions& options_tidl_onnx_vec)
      : options_tidl_onnx_vec_(options_tidl_onnx_vec), type_(type) {}
  ~TidlProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

  private:
  TIDLProviderOptions options_tidl_onnx_vec_;
  std::string type_;
};

std::unique_ptr<IExecutionProvider> TidlProviderFactory::CreateProvider() {
  //return onnxruntime::make_unique<TidlExecutionProvider>();
  TidlExecutionProviderInfo info(type_, options_tidl_onnx_vec_);
  return onnxruntime::make_unique<TidlExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tidl(const std::string &type, const TIDLProviderOptions& options_tidl_onnx_vec) {
  return std::make_shared<onnxruntime::TidlProviderFactory>(type, options_tidl_onnx_vec);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionsOptionsSetDefault_Tidl, _In_ c_api_tidl_options * options_tidl_onnx) {

  memset(options_tidl_onnx, 0, sizeof(c_api_tidl_options));
  options_tidl_onnx->debug_level = 0;
  options_tidl_onnx->priority = 0;
  options_tidl_onnx->max_pre_empt_delay = FLT_MAX;
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Tidl, _In_ OrtSessionOptions* options, c_api_tidl_options * options_tidl_onnx) {
  TIDLProviderOptions options_tidl_onnx_vec;

  options_tidl_onnx_vec.push_back(std::make_pair("debug_level", std::to_string(options_tidl_onnx->debug_level)));
  options_tidl_onnx_vec.push_back(std::make_pair("priority", std::to_string(options_tidl_onnx->priority)));
  options_tidl_onnx_vec.push_back(std::make_pair("max_pre_empt_delay", std::to_string(options_tidl_onnx->max_pre_empt_delay)));
  options_tidl_onnx_vec.push_back(std::make_pair("artifacts_folder", std::string(options_tidl_onnx->artifacts_folder)));

  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Tidl("", options_tidl_onnx_vec));
  return nullptr;
}
