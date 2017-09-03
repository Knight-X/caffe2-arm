
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
  class ARMSoftmaxOp  final : public Operator<CPUContext> {
    public:
      ARMSoftmaxOp(const OperatorDef& operator_def, Workspace* ws)
	      : Operator<CPUContext>(operator_def, ws) {
        auto& X = Input(0);
	auto* Y = Output(0);
	const unsigned int N = X.dim32(0), W = X.dim32(1);
	arm_compute::TensorShape src_shape(static_cast<unsigned int>(X.size()));

	src.allocator()->init(arm_compute::TensorInfo(src_shape, 1, arm_compute::DataType::F32));
	res.allocator()->init(arm_compute::TensorInfo(src_shape, 1, arm_compute::DataType::F32));
	softmax.configure(&src, &res);
	src.allocator()->allocate();
	res.allocator()->allocate();
	}
    void fillsrc(arm_compute::Tensor &tensor, const float* src, int N, int W) {
      arm_compute::Window window;
      window.set(0, arm_compute::Window::Dimension(0, N * W, 1));
      
      arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates &id) {
        float value = src[id.x()];
	void *out = tensor.ptr_to_element(id);
	*reinterpret_cast<float *>(out) = value;
      });
    }
    void filldst(arm_compute::Tensor &tensor, float* dst, int N, int W) {
      arm_compute::Window window;
      window.set(0, arm_compute::Window::Dimension(0, N * W, 1));
      
      arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates &id) {
	void *out = tensor.ptr_to_element(id);
        dst[id.x()] = *reinterpret_cast<float *>(out);
      });

    }

      bool RunOnDevice() override {
        auto& X = Input(0);
	auto* Y = Output(0);
	int N = X.dim32(0), W = X.dim32(1);
	const float *_src = X.template data<float>();
	fillsrc(src, _src, N, W); 

	softmax.run();	

	float *_res = Y->template mutable_data<float>();
	filldst(res, _res, N, W);
	return true;

      }
    private:
      arm_compute::Tensor src;
      arm_compute::Tensor res;
      arm_compute::NESoftmaxLayer softmax;

  }; 
  REGISTER_CPU_OPERATOR_WITH_ENGINE(Softmax, ARM, ARMSoftmaxOp);
}
