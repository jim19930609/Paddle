#include "paddle/pten/hapi/all.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/eager/function_api.h"
#include "paddle/fluid/eager/nodes/matmul_v2_node.h"

std::vector<std::vector<egr::EagerTensor>> GradNodeMatmul::operator()(const std::vector<std::vector<egr::EagerTensor>>& grads) {
  
  // 1. Check Output Size
  PADDLE_ENFORCE(((grads.size() == 1) && (grads[0].size() == 1)),
                 paddle::platform::errors::Fatal(
                     "MatmulGradNode should take exactly 1 grad tensor"
                     "However received: %d",
                     grads.size()));

  egr::EagerTensor grad_x;
  egr::EagerTensor grad_y;

  // 2. Create needed out parttern
  // Apply Gradient Hooks
  if (GradientHooksRegistered()) {
    // TODO(jiabin): Shall we apply hook slot by slot here or accept
    // vector<vector<pten::tensor>> to apply all hooks?
    std::vector<std::vector<egr::EagerTensor>> hooked_grads =
        ApplyGradientHooks(grads);
    
    const std::shared_ptr<paddle::experimental::Tensor>& x_tensor = X_.recover(nullptr).Tensor();
    const std::shared_ptr<paddle::experimental::Tensor>& y_tensor = Y_.recover(nullptr).Tensor();
    const std::shared_ptr<paddle::experimental::Tensor>& hooked_grad = hooked_grads[0][0].Tensor();
    PADDLE_ENFORCE(x_tensor != nullptr,
                 paddle::platform::errors::Fatal(
                     "Underlying member \"tensor_\" of InputWrapper X is Null"));
    PADDLE_ENFORCE(y_tensor != nullptr,
                 paddle::platform::errors::Fatal(
                     "Underlying member \"tensor_\" of InputWrapper Y is Null"));
    PADDLE_ENFORCE(hooked_grad != nullptr,
                 paddle::platform::errors::Fatal(
                     "Underlying member \"tensor_\" of hooked_grad is Null"));

    std::vector<paddle::experimental::Tensor> grad_tensors = paddle::experimental::grad_matmul(*x_tensor.get(), *y_tensor.get(), *hooked_grad.get(), transpose_x_, transpose_y_);
    
    grad_x.set_tensor(std::make_shared<paddle::experimental::Tensor>(grad_tensors[0]));
    grad_y.set_tensor(std::make_shared<paddle::experimental::Tensor>(grad_tensors[1]));

  } else {
    const std::shared_ptr<paddle::experimental::Tensor>& x_tensor = X_.recover(nullptr).Tensor();
    const std::shared_ptr<paddle::experimental::Tensor>& y_tensor = Y_.recover(nullptr).Tensor();
    const std::shared_ptr<paddle::experimental::Tensor>& grad = grads[0][0].Tensor();
    PADDLE_ENFORCE(x_tensor != nullptr,
                 paddle::platform::errors::Fatal(
                     "Underlying member \"tensor_\" of InputWrapper X is Null"));
    PADDLE_ENFORCE(y_tensor != nullptr,
                 paddle::platform::errors::Fatal(
                     "Underlying member \"tensor_\" of InputWrapper Y is Null"));
    PADDLE_ENFORCE(grad != nullptr,
                 paddle::platform::errors::Fatal(
                     "Underlying member \"tensor_\" of grad is Null"));

    std::vector<paddle::experimental::Tensor> grad_tensors = paddle::experimental::grad_matmul(*x_tensor.get(), *y_tensor.get(), *grad.get(), transpose_x_, transpose_y_);
    
    grad_x.set_tensor(std::make_shared<paddle::experimental::Tensor>(grad_tensors[0]));
    grad_y.set_tensor(std::make_shared<paddle::experimental::Tensor>(grad_tensors[1]));
  }

  // Apply Reduce Hooks
  if (ReduceHooksRegistered()) {
    ApplyReduceHooks();
  }

  return {{grad_x}, {grad_y}};

}
