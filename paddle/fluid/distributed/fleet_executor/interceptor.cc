// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/fleet_executor/interceptor.h"
#include "paddle/fluid/distributed/fleet_executor/carrier.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"

namespace paddle {
namespace distributed {

Interceptor::Interceptor(int64_t interceptor_id, TaskNode* node)
    : interceptor_id_(interceptor_id), node_(node) {
  interceptor_thread_ = std::thread([this]() {
    VLOG(3) << "Interceptor " << interceptor_id_
            << " starts the thread pooling it's local mailbox.";
    PoolTheMailbox();
  });
}

Interceptor::~Interceptor() { Join(); }

void Interceptor::Join() {
  if (interceptor_thread_.joinable()) {
    interceptor_thread_.join();
  }
}

void Interceptor::RegisterMsgHandle(MsgHandle handle) { handle_ = handle; }

void Interceptor::Handle(const InterceptorMessage& msg) {
  PADDLE_ENFORCE_NOT_NULL(handle_, platform::errors::PreconditionNotMet(
                                       "Message handle is not registered."));
  handle_(msg);
}

void Interceptor::StopCarrier() {
  PADDLE_ENFORCE_NOT_NULL(carrier_, platform::errors::PreconditionNotMet(
                                        "Carrier is not registered."));
  std::condition_variable& cond_var = carrier_->GetCondVar();
  // probably double notify, but ok for ut
  cond_var.notify_all();
}

int64_t Interceptor::GetInterceptorId() const {
  // return the interceptor id
  return interceptor_id_;
}

void Interceptor::EnqueueRemoteInterceptorMessage(
    const InterceptorMessage& interceptor_message) {
  // Called by Carrier, enqueue an InterceptorMessage to remote mailbox
  VLOG(3) << "Enqueue message: " << interceptor_message.message_type()
          << " into " << interceptor_id_ << "'s remote mailbox.";
  remote_mailbox_.Push(interceptor_message);
}

bool Interceptor::Send(int64_t dst_id, InterceptorMessage& msg) {
  PADDLE_ENFORCE_NOT_NULL(carrier_, platform::errors::PreconditionNotMet(
                                        "Carrier is not registered."));
  msg.set_src_id(interceptor_id_);
  msg.set_dst_id(dst_id);
  return carrier_->Send(msg);
}

void Interceptor::PoolTheMailbox() {
  // pool the local mailbox, parse the Message
  for (;;) {
    if (local_mailbox_.empty()) {
      // local mailbox is empty, fetch the remote mailbox
      VLOG(3) << interceptor_id_ << "'s local mailbox is empty. "
              << "Fetch the remote mailbox.";
      PADDLE_ENFORCE_EQ(FetchRemoteMailbox(), true,
                        platform::errors::InvalidArgument(
                            "Error encountered when fetch remote mailbox."));
    }
    const InterceptorMessage interceptor_message = local_mailbox_.front();
    local_mailbox_.pop_front();
    const MessageType message_type = interceptor_message.message_type();
    VLOG(3) << "Interceptor " << interceptor_id_ << " has received a message"
            << " from interceptor " << interceptor_message.src_id()
            << " with message: " << message_type << ".";

    Handle(interceptor_message);

    if (stop_) {
      // break the pooling thread
      VLOG(3) << "Interceptor " << interceptor_id_ << " is quiting.";
      break;
    }
  }
}

bool Interceptor::FetchRemoteMailbox() {
  remote_mailbox_.PopAll(&local_mailbox_);
  return !local_mailbox_.empty();
}

static InterceptorFactory::CreateInterceptorMap& GetInterceptorMap() {
  static InterceptorFactory::CreateInterceptorMap interceptorMap;
  return interceptorMap;
}

std::unique_ptr<Interceptor> InterceptorFactory::Create(const std::string& type,
                                                        int64_t id,
                                                        TaskNode* node) {
  auto& interceptor_map = GetInterceptorMap();
  auto iter = interceptor_map.find(type);
  PADDLE_ENFORCE_NE(
      iter, interceptor_map.end(),
      platform::errors::NotFound("interceptor %s is not register", type));
  return iter->second(id, node);
}

void InterceptorFactory::Register(
    const std::string& type, InterceptorFactory::CreateInterceptorFunc func) {
  auto& interceptor_map = GetInterceptorMap();
  interceptor_map.emplace(type, func);
}

}  // namespace distributed
}  // namespace paddle
