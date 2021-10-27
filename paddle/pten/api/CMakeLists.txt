set(PTEN_DEPS convert_utils dense_tensor kernel_factory kernel_context)
set(PTEN_DEPS ${PTEN_DEPS} math_cpu linalg_cpu creation_cpu manipulation_cpu reduce_cpu grad_linalg_cpu grad_reduce_cpu)
set(PTEN_DEPS ${PTEN_DEPS} unary binary)
if(WITH_GPU OR WITH_ROCM)
  set(PTEN_DEPS ${PTEN_DEPS} math_cuda linalg_cuda creation_cuda manipulation_cuda reduce_cuda grad_linalg_cuda grad_reduce_cuda)
endif()

cc_library(pten SRCS all.cc DEPS ${PTEN_DEPS})
