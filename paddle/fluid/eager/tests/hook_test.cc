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

#include <sstream>

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "paddle/fluid/eager/nodes/accumulation_node.h"
#include "paddle/fluid/eager/nodes/scale_node.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/autograd_meta.h"

#include "paddle/fluid/eager/api/api.h"

#include "paddle/top/core/tensor_meta.h"
#include "paddle/top/core/dense_tensor.h"

pt::Tensor hook_function(const pt::Tensor& t) { 
    auto t_dense = std::dynamic_pointer_cast<pt::DenseTensor>(t.impl());
    
    auto ret_meta = std::make_unique<pt::TensorMeta>(t_dense->dims(), t_dense->backend(), t_dense->type(), t_dense->layout());
    auto ret_dense = std::make_shared<pt::DenseTensor>(std::move(ret_meta));
    
    float* t_ptr = t_dense->mutable_data<float>();
    float* ret_ptr = ret_dense->mutable_data<float>();
    for(int i = 0; i < ret_dense->numel(); i++) {
        ret_ptr[i] = t_ptr[i] + 3.0;
    }
    
    auto ret_impl = std::dynamic_pointer_cast<pt::TensorInterface>(ret_dense);
    pt::Tensor ret = pt::Tensor();
    ret.SetImpl(ret_impl);
    
    return ret;
};

/*
AccumulationNode
  |
  |retain_grad
  |hook
  |
ScaleNode
  |
  |retain_grad
  |hook
  |
 inp0
*/
TEST(RetainGrad, HookBeforeRetainGrad) {
  // Create Target Tensor
  // Use Empty Grad Tensor
  std::vector<pt::Tensor> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  {
      auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
      auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
      auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
      
      auto tensor = pt::Tensor(tensor_impl);
      target_tensors.emplace_back(std::move(tensor)); 
  }
  pt::Tensor& target_tensor = target_tensors[0];
  
  // Create ScaleNode
  auto scale_node_ptr = std::make_shared<egr::GradNodeScale>();
  scale_node_ptr->SetAttributes(5.0/*scale*/);
  
  // Create AccumulationNode
  auto acc_node_ptr = std::make_shared<egr::GradNodeAccumulation>();

  // Connect Input Tensor and ScaleNode via AutoGradMeta
  // Apply RetainGrad
  {
      // ScaleNode Hook: +3
      std::function<pt::Tensor(const pt::Tensor&)> hook = &hook_function;

      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(scale_node_ptr));
      auto_grad_meta->SetOutRank(0);
      target_tensor.SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
      
      egr::RegisterGradientHookForTensor(target_tensor, hook);
      egr::RetainGradForTensor(target_tensor); // result: 1.0 + 3.0 = 4.0
  }

  // Connect ScaleNode -> AccumulationNode via Edge
  {
      auto meta = egr::AutogradMeta();
      meta.SetOutRank(0);
      meta.SetGradNode(acc_node_ptr);
      scale_node_ptr->AddEdges({ &meta });
  }
  
  // Retain Grad for leaf tensor1
  pt::Tensor leaf_tensor = pt::Tensor();
  {
      // AccumulationNode Hook: +3
      std::function<pt::Tensor(const pt::Tensor&)> hook = &hook_function;
      
      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(acc_node_ptr));
      auto_grad_meta->SetOutRank(0);
      leaf_tensor.SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
      
      egr::RegisterGradientHookForTensor(leaf_tensor, hook);
      egr::RetainGradForTensor(leaf_tensor); // result: 4.0*5.0 + 3.0 = 23.0
  }

  // Use Empty Grad Tensor
  egr::RunBackward(target_tensors, {});

  // Print target tensor grad
  {
      egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(target_tensor);
      auto target_grad_dense = std::dynamic_pointer_cast<pt::DenseTensor>(meta->Grad().impl());
      float* ptr = target_grad_dense->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 4.0, 
            paddle::platform::errors::Fatal("Numerical Error, Expected %f but got %f", 4.0, ptr[i]));
      }
  }

  // Print leaf tensor grad
  {
      egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(leaf_tensor);
      auto leaf_grad_dense = std::dynamic_pointer_cast<pt::DenseTensor>(meta->Grad().impl());
      float* ptr = leaf_grad_dense->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 23.0, 
            paddle::platform::errors::Fatal("Numerical Error, Expected %f but got %f", 23.0, ptr[i]));
      }
  }
}

/*
AccumulationNode
  |
  |hook
  |retain_grad
  |
ScaleNode
  |
  |hook
  |retain_grad
  |
 inp0
*/
TEST(RetainGrad, HookAfterRetainGrad) {
  // Create Target Tensor
  // Use Empty Grad Tensor
  std::vector<pt::Tensor> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  {
      auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
      auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
      auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
      
      auto tensor = pt::Tensor(tensor_impl);
      target_tensors.emplace_back(std::move(tensor)); 
  }
  pt::Tensor& target_tensor = target_tensors[0];
  
  // Create ScaleNode
  auto scale_node_ptr = std::make_shared<egr::GradNodeScale>();
  scale_node_ptr->SetAttributes(5.0/*scale*/);
  
  // Create AccumulationNode
  auto acc_node_ptr = std::make_shared<egr::GradNodeAccumulation>();

  // Connect Input Tensor and ScaleNode via AutoGradMeta
  // Apply RetainGrad
  {
      // ScaleNode Hook: +3
      std::function<pt::Tensor(const pt::Tensor&)> hook = &hook_function;

      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(scale_node_ptr));
      auto_grad_meta->SetOutRank(0);
      target_tensor.SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
      
      egr::RetainGradForTensor(target_tensor); // result: 1.0
      egr::RegisterGradientHookForTensor(target_tensor, hook);
  }

  // Connect ScaleNode -> AccumulationNode via Edge
  {
      auto meta = egr::AutogradMeta();
      meta.SetOutRank(0);
      meta.SetGradNode(acc_node_ptr);
      scale_node_ptr->AddEdges({ &meta });
  }
  
  // Retain Grad for leaf tensor1
  pt::Tensor leaf_tensor = pt::Tensor();
  {
      // AccumulationNode Hook: +3
      std::function<pt::Tensor(const pt::Tensor&)> hook = &hook_function;
      
      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(acc_node_ptr));
      auto_grad_meta->SetOutRank(0);
      leaf_tensor.SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
      
      egr::RetainGradForTensor(leaf_tensor); // RetainGrad for leaf tensor gets postponed, result: 4.0*5.0 + 3.0 = 23.0
      egr::RegisterGradientHookForTensor(leaf_tensor, hook);
  }

  // Use Empty Grad Tensor
  egr::RunBackward(target_tensors, {});

  // Print target tensor grad
  {
      egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(target_tensor);
      auto target_grad_dense = std::dynamic_pointer_cast<pt::DenseTensor>(meta->Grad().impl());
      float* ptr = target_grad_dense->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 1.0, 
            paddle::platform::errors::Fatal("Numerical Error, Expected %f but got %f", 1.0, ptr[i]));
      }
  }

  // Print leaf tensor grad
  {
      egr::AutogradMeta* meta = egr::EagerUtils::autograd_meta(leaf_tensor);
      auto leaf_grad_dense = std::dynamic_pointer_cast<pt::DenseTensor>(meta->Grad().impl());
      float* ptr = leaf_grad_dense->mutable_data<float>();
      for(int i = 0; i < 20; i++) {
          PADDLE_ENFORCE(ptr[i] == 23.0, 
            paddle::platform::errors::Fatal("Numerical Error, Expected %f but got %f", 23.0, ptr[i]));
      }
  }
}

TEST(GradientHook, SingleNode) {
  // Create Target Tensor
  // Use Empty Grad Tensor
  std::vector<pt::Tensor> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  {
      auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
      auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
      auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
      
      auto tensor = pt::Tensor(tensor_impl);
      target_tensors.emplace_back(std::move(tensor)); 
  }
  
  // Create Scale Node
  auto node0_ptr = std::make_shared<egr::GradNodeScale>();
  node0_ptr->SetAttributes(5.0/*scale*/);

  // Connect Tensor and Node via AutoGradMeta
  {
      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(node0_ptr));
      auto_grad_meta->SetOutRank(0);
      target_tensors[0].SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
  }

  // Register GradientHook
  std::function<pt::Tensor(const pt::Tensor&)> hook = [](const pt::Tensor& t) { 
    pt::Tensor ret = pt::Tensor();

    // Copy t::impl()
    ret = t;
    auto ret_dense = std::dynamic_pointer_cast<pt::DenseTensor>(ret.impl());
    float* ret_ptr = ret_dense->mutable_data<float>();

    for(int i = 0; i < ret_dense->numel(); i++) {
        ret_ptr[i] += 2.0;
    }

    return ret;
  };
  egr::RegisterGradientHookForTensor(target_tensors[0], hook);

  // Use Empty Grad Tensor
  egr::RunBackward(target_tensors, {});

  // result should be: (1.0 + 2.0) * 5.0 = 15.0
}

/*
Node1
  |
Node0
  |
 inp0
*/
TEST(GradientHook, LinearNodes) {
  // Create Target Tensor
  // Use Empty Grad Tensor
  std::vector<pt::Tensor> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  {
      auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
      auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
      auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
      
      auto tensor = pt::Tensor(tensor_impl);
      target_tensors.emplace_back(std::move(tensor)); 
  }
  
  // Create Node0
  auto node0_ptr = std::make_shared<egr::GradNodeScale>();
  node0_ptr->SetAttributes(5.0/*scale*/);
  
  // Create Node1
  auto node1_ptr = std::make_shared<egr::GradNodeScale>();
  node1_ptr->SetAttributes(10.0/*scale*/);

  // Connect Input Tensor and Node0 via AutoGradMeta
  {
      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(node0_ptr));
      auto_grad_meta->SetOutRank(0);
      target_tensors[0].SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
  }

  // Connect Node0 -> Node1 via Edge
  {
      auto meta = egr::AutogradMeta();
      meta.SetOutRank(0);
      meta.SetGradNode(node1_ptr);
      node0_ptr->AddEdges({ &meta });
  }
  
  // Register Hooks
  {
      // Node0 Hook
      std::function<pt::Tensor(const pt::Tensor&)> hook0 = [](const pt::Tensor& t) { 
        pt::Tensor ret = pt::Tensor();

        // Copy t::impl()
        ret = t;
        auto ret_dense = std::dynamic_pointer_cast<pt::DenseTensor>(ret.impl());
        float* ret_ptr = ret_dense->mutable_data<float>();

        for(int i = 0; i < ret_dense->numel(); i++) {
            ret_ptr[i] += 1.0;
        }
        return ret;
      };
      egr::RegisterGradientHookForTensor(target_tensors[0], hook0);
      
      // Node1 Hook
      std::function<pt::Tensor(const pt::Tensor&)> hook1 = [](const pt::Tensor& t) { 
        pt::Tensor ret = pt::Tensor();

        // Copy t::impl()
        ret = t;
        auto ret_dense = std::dynamic_pointer_cast<pt::DenseTensor>(ret.impl());
        float* ret_ptr = ret_dense->mutable_data<float>();

        for(int i = 0; i < ret_dense->numel(); i++) {
            ret_ptr[i] += 3.0;
        }
        return ret;
      };
      // Fake an AutogradMeta
      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(node1_ptr));
      auto_grad_meta->SetOutRank(0);
      pt::Tensor fake_tensor = pt::Tensor();
      fake_tensor.SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
      egr::RegisterGradientHookForTensor(fake_tensor, hook1);
  }
  // Use Empty Grad Tensor
  egr::RunBackward(target_tensors, {});

  // result: ((1.0+1.0)*5.0 + 3.0)*10.0 = 130.0
}


/*
    Node2
    |   |
Node0   Node1
  |      |
 inp0   inp1
*/
TEST(GradientHook, WithAccumulation) {
  // Create Target Tensor
  std::vector<pt::Tensor> target_tensors;
  paddle::framework::DDim ddim = paddle::framework::make_ddim({4, 16, 16, 32});
  // inp0
  {
      auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
      auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
      auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
      
      auto tensor = pt::Tensor(tensor_impl);
      target_tensors.emplace_back(std::move(tensor)); 
  }

  // inp1
  {
      auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
      auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
      auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
      
      auto tensor = pt::Tensor(tensor_impl);
      target_tensors.emplace_back(std::move(tensor)); 
  }
  
  // Create Grad Tensor
  std::vector<pt::Tensor> grad_tensors;
  
  // inp0
  {
      auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
      auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
      
      float* ptr = tensor_dense->mutable_data<float>();
      for(int i = 0; i < tensor_dense->numel(); i++) {
          ptr[i] = 5.0;
      }
      
      auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
      auto tensor = pt::Tensor(tensor_impl);
      grad_tensors.emplace_back(std::move(tensor)); 
  }

  // inp1
  {
      auto tensor_meta = std::make_unique<pt::TensorMeta>(ddim, pt::Backend::kCPU, 
              pt::DataType::kFLOAT32, pt::DataLayout::kNCHW);
      auto tensor_dense = std::make_shared<pt::DenseTensor>(std::move(tensor_meta));
      
      float* ptr = tensor_dense->mutable_data<float>();
      for(int i = 0; i < tensor_dense->numel(); i++) {
          ptr[i] = 10.0;
      }
      
      auto tensor_impl = std::dynamic_pointer_cast<pt::TensorInterface>(tensor_dense);
      auto tensor = pt::Tensor(tensor_impl);
      grad_tensors.emplace_back(std::move(tensor)); 
  }
  
  // Create Node0
  auto node0_ptr = std::make_shared<egr::GradNodeScale>();
  node0_ptr->SetAttributes(5.0/*scale*/);
  
  // Create Node1
  auto node1_ptr = std::make_shared<egr::GradNodeScale>();
  node1_ptr->SetAttributes(10.0/*scale*/);
  
  // Create Node2
  auto node2_ptr = std::make_shared<egr::GradNodeScale>();
  node2_ptr->SetAttributes(20.0/*scale*/);

  // Connect Inp0 and Node0 via AutoGradMeta
  {
      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(node0_ptr));
      auto_grad_meta->SetOutRank(0);
      target_tensors[0].SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
  }
  
  // Connect Inp1 and Node1 via AutoGradMeta
  {
      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(node1_ptr));
      auto_grad_meta->SetOutRank(0);
      target_tensors[1].SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
  }

  // Connect Node0 -> Node2 via Edge
  {
      auto meta = egr::AutogradMeta();
      meta.SetOutRank(0);
      meta.SetGradNode(node2_ptr);
      node0_ptr->AddEdges({ &meta });
  }
  
  // Connect Node1 -> Node2 via Edge
  {
      auto meta = egr::AutogradMeta();
      meta.SetOutRank(0);
      meta.SetGradNode(node2_ptr);
      node1_ptr->AddEdges({ &meta });
  }

  // Register Hooks
  {
      // Node0 Hook
      std::function<pt::Tensor(const pt::Tensor&)> hook0 = [](const pt::Tensor& t) { 
        pt::Tensor ret = pt::Tensor();
        ret = t;
        auto ret_dense = std::dynamic_pointer_cast<pt::DenseTensor>(ret.impl());
        float* ret_ptr = ret_dense->mutable_data<float>();

        for(int i = 0; i < ret_dense->numel(); i++) {
            ret_ptr[i] += 1.0;
        }
        return ret;
      };
      egr::RegisterGradientHookForTensor(target_tensors[0], hook0);
      
      // Node1 Hook
      std::function<pt::Tensor(const pt::Tensor&)> hook1 = [](const pt::Tensor& t) { 
        pt::Tensor ret = pt::Tensor();
        ret = t;
        auto ret_dense = std::dynamic_pointer_cast<pt::DenseTensor>(ret.impl());
        float* ret_ptr = ret_dense->mutable_data<float>();

        for(int i = 0; i < ret_dense->numel(); i++) {
            ret_ptr[i] += 2.0;
        }
        return ret;
      };
      egr::RegisterGradientHookForTensor(target_tensors[1], hook1);
      
      // Node2 Hook
      std::function<pt::Tensor(const pt::Tensor&)> hook2 = [](const pt::Tensor& t) { 
        pt::Tensor ret = pt::Tensor();
        ret = t;
        auto ret_dense = std::dynamic_pointer_cast<pt::DenseTensor>(ret.impl());
        float* ret_ptr = ret_dense->mutable_data<float>();

        for(int i = 0; i < ret_dense->numel(); i++) {
            ret_ptr[i] += 3.0;
        }
        return ret;
      };

      // Fake an AutogradMeta
      auto auto_grad_meta = std::make_shared<egr::AutogradMeta>();
      auto_grad_meta->SetGradNode(std::dynamic_pointer_cast<egr::GradNodeBase>(node2_ptr));
      auto_grad_meta->SetOutRank(0);
      pt::Tensor fake_tensor = pt::Tensor();
      fake_tensor.SetAutoGradMeta(std::dynamic_pointer_cast<pt::AbstractAutogradMeta>(auto_grad_meta));
      egr::RegisterGradientHookForTensor(fake_tensor, hook2);
  }

  // Use Empty Grad Tensor
  egr::RunBackward(target_tensors, grad_tensors);

  // result: ((5+1)*5 + (10+2)*10 + 3) * 20 = 3060
}
