#pragma once
// Input value checks for composite native entry points.
//
// Eager calls raise ordinary synchronous errors, so invalid CUDA data cannot
// trigger a device-side assert that poisons the CUDA context. Tracing (Fake,
// meta and functional tensors or Python dispatch modes) and CUDA stream
// capture keep graph-safe asynchronous assertions instead.
#include <ATen/ATen.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/VirtualGuardImpl.h>

#include <cstring>
#include <utility>
#include <vector>

namespace torchpairwise::ops::utils {
using CheckList = std::vector<std::pair<at::Tensor, const char*>>;

inline bool defers_value_checks(const at::Tensor& condition) {
  constexpr auto traced = c10::DispatchKeySet(
      {c10::DispatchKey::Python, c10::DispatchKey::PythonTLSSnapshot,
       c10::DispatchKey::Functionalize});
  if (condition.is_meta() || condition.key_set().has_any(traced) ||
      c10::impl::tls_local_dispatch_key_set().included_.has_any(traced))
    return true;
  const auto device = condition.device();
  return device.is_cuda() && c10::impl::VirtualGuardImpl(device.type())
                                 .getStream(device)
                                 .is_capturing();
}

namespace detail {
inline CheckList*& active_deferred_checks() {
  thread_local CheckList* active = nullptr;
  return active;
}
// Raises the first failing message; at most one synchronization when valid.
inline void evaluate_checks(const CheckList& checks) {
  if (checks.empty())
    return;
  std::vector<at::Tensor> conditions;
  for (const auto& check : checks)
    conditions.push_back(
        check.first.reshape({}).to(checks.front().first.device()));
  if (!at::stack(conditions).all().item<bool>())
    for (const auto& check : checks)
      TORCH_CHECK(check.first.item<bool>(), check.second);
}
}  // namespace detail

// Collects Boolean scalar conditions for one native call.
class ValueChecks {
 public:
  void require(at::Tensor condition, const char* message) {
    checks_.emplace_back(std::move(condition), message);
  }
  void finish() {
    bool deferred = false;
    for (const auto& check : checks_)
      deferred = deferred || defers_value_checks(check.first);
    if (deferred) {
      for (const auto& check : checks_)
        at::_assert_async(check.first, check.second);
    } else if (auto* pending = detail::active_deferred_checks()) {
      // One running conjunction per message keeps tile loops bounded.
      for (auto& check : checks_) {
        auto condition = check.first.reshape({});
        bool merged = false;
        for (auto& item : *pending)
          if (std::strcmp(item.second, check.second) == 0) {
            item.first = item.first & condition.to(item.first.device());
            merged = true;
            break;
          }
        if (!merged)
          pending->emplace_back(condition, check.second);
      }
    } else {
      detail::evaluate_checks(checks_);
    }
    checks_.clear();
  }

 private:
  CheckList checks_;
};

inline void check_values(const at::Tensor& condition, const char* message) {
  ValueChecks checks;
  checks.require(condition, message);
  checks.finish();
}

// Tiled kernels call composite entry points once per tile. Within this
// scope, eager checks accumulate on the device and finish() synchronizes once
// for the whole kernel. Nested scopes defer to the outermost one.
class DeferredValueChecks {
 public:
  DeferredValueChecks() : owner_(detail::active_deferred_checks() == nullptr) {
    if (owner_)
      detail::active_deferred_checks() = &checks_;
  }
  DeferredValueChecks(const DeferredValueChecks&) = delete;
  DeferredValueChecks& operator=(const DeferredValueChecks&) = delete;
  ~DeferredValueChecks() {
    if (owner_ && detail::active_deferred_checks() == &checks_)
      detail::active_deferred_checks() = nullptr;
  }
  void finish() {
    if (!owner_)
      return;
    detail::active_deferred_checks() = nullptr;
    detail::evaluate_checks(checks_);
    checks_.clear();
  }

 private:
  bool owner_;
  CheckList checks_;
};
}  // namespace torchpairwise::ops::utils
