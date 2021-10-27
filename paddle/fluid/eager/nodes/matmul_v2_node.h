#pragma once
#include "paddle/fluid/eager/tensor_wrapper.h"
#include "paddle/fluid/eager/grad_node_info.h"

class GradNodeMatmul : public egr::GradNodeBase {
 public:
  GradNodeMatmul() : egr::GradNodeBase() {}
  GradNodeMatmul(size_t bwd_in_slot_num, size_t bwd_out_slot_num) : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}
  ~GradNodeMatmul() override = default;

  virtual std::vector<std::vector<egr::EagerTensor>> operator()(const std::vector<std::vector<egr::EagerTensor>>& grads) override;

  // SetX, SetY, ...
   void SetTensorWrapperX(const egr::EagerTensor& X) {
     X_ = egr::TensorWrapper(X, true /*full_reserved*/);
   }
   void SetTensorWrapperY(const egr::EagerTensor& Y) {
     Y_ = egr::TensorWrapper(Y, true /*full_reserved*/);
   }

  // SetAttr0, SetAttr1, ...
   void SetAttributes_transpose_x(const bool transpose_x) {
     transpose_x_ = transpose_x;
   }
   
   void SetAttributes_transpose_y(const bool transpose_y) {
     transpose_y_ = transpose_y;
   }

 private:
   // TensorWrappers
   egr::TensorWrapper X_;
   egr::TensorWrapper Y_;

   // Attribute Members
   bool transpose_x_ = 0;
   bool transpose_y_ = 0;

};
