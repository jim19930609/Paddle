add_subdirectory(utils)

cc_library(math_api SRCS math.cc DEPS pten)
cc_library(linalg_api SRCS linalg.cc DEPS pten)
cc_library(creation_api SRCS creation.cc DEPS pten)
cc_library(manipulation_api SRCS manipulation.cc DEPS pten)

cc_library(reduce_api SRCS reduce.cc DEPS pten)
cc_library(grad_linalg_api SRCS grad_linalg.cc DEPS pten)
cc_library(grad_reduce_api SRCS grad_reduce.cc DEPS pten)
