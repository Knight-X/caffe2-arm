
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/operators/leaky_relu_op.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/runtime/Tensor.h"

namespace caffe2 {
  class ARMFullyConnectedOp  final : public Operator<CPUContext> {
    public:
      ARMFullyConnectedOp(const OperatorDef& operator_def, Workspace* ws)
	      : Operator<CPUContext>(operator_def, ws) {
	      }
    void fillsrc(arm_compute::Tensor &tensor, const float* src, int M, int K) {
      arm_compute::Window window;
      window.set(0, arm_compute::Window::Dimension(0, K, 1));
      
      arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates &id) {
        float value = src[id.x()];
	void *out = tensor.ptr_to_element(id);
	*reinterpret_cast<float *>(out) = value;
      });
    }
    void fillweight(arm_compute::Tensor &tensor, const float* src, int K, int N) {
      arm_compute::Window window;
      window.set(0, arm_compute::Window::Dimension(0, K, 1));
      window.set(1, arm_compute::Window::Dimension(0, N, 1));
      
      arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates &id) {
        float value = src[id.y() * K + id.x()];
	void *out = tensor.ptr_to_element(id);
	*reinterpret_cast<float *>(out) = value;
      });
    }
    void fillbias(arm_compute::Tensor &tensor, const float* src, int N) {
      arm_compute::Window window;
      window.set(0, arm_compute::Window::Dimension(0, N, 1));
      
      arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates &id) {
        float value = src[id.x()];
	void *out = tensor.ptr_to_element(id);
	*reinterpret_cast<float *>(out) = value;
      });
    }
    void filldst(arm_compute::Tensor &tensor, float* dst, int M, int N) {
      arm_compute::Window window;
      window.set(0, arm_compute::Window::Dimension(0, N, 1));
      
      arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates &id) {
	void *out = tensor.ptr_to_element(id);
        dst[id.x()] = *reinterpret_cast<float *>(out);
      });

    }

      bool RunOnDevice() override {
        auto& X = Input(0);
	auto& W = Input(1);
	auto& b = Input(2);
	auto* Y = Output(0);
	const int M = X.dim32(0), K = X.dim32(1);
	const int N = W.dim32(0);
	if (W.dim32(0) == K) {
	  std::cout << "equal " << std::endl;
	}
	arm_compute::Tensor src;
	arm_compute::TensorShape src_shape(K);
	arm_compute::TensorShape weight_shape(K, N);
	arm_compute::TensorShape bias_shape(N);
	arm_compute::TensorShape res_shape(N);
	arm_compute::Tensor weight;
	arm_compute::Tensor bias;
	arm_compute::Tensor res;
	arm_compute::NEFullyConnectedLayer fc;

	src.allocator()->init(arm_compute::TensorInfo(src_shape, 1, arm_compute::DataType::F32));
	weight.allocator()->init(arm_compute::TensorInfo(weight_shape, 1, arm_compute::DataType::F32));
	bias.allocator()->init(arm_compute::TensorInfo(bias_shape, 1, arm_compute::DataType::F32));
	res.allocator()->init(arm_compute::TensorInfo(res_shape, 1, arm_compute::DataType::F32));
	fc.configure(&src, &weight, &bias, &res);
	src.allocator()->allocate();
	weight.allocator()->allocate();
	bias.allocator()->allocate();
	res.allocator()->allocate();

	const float *_src = X.template data<float>();
	const float *_bias = b.template data<float>();
	const float *_weight = W.template data<float>();
	fillsrc(src, _src, M, K); 
	fillweight(weight,_weight, K, N);
	fillbias(bias, _bias, N);
	void *src_o = src.ptr_to_element(arm_compute::Coordinates(0, 0));
	float tmp_o = *reinterpret_cast<float *>(src_o);
	std::cout << "finish fc " << tmp_o << " ";

	fc.run();	
	void *out = res.ptr_to_element(arm_compute::Coordinates(0, 0));
	float tmp = *reinterpret_cast<float *>(out);
	std::cout << "finish fc " << tmp << " ";

	float *_res = Y->template mutable_data<float>();
	filldst(res, _res, M, N);

      }
  }; 
  REGISTER_CPU_OPERATOR_WITH_ENGINE(FC, ARM, ARMFullyConnectedOp);
}
