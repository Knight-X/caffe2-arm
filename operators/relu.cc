
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
  class ARMReluOp  final : public Operator<CPUContext> {
    public:
      ARMReluOp(const OperatorDef& operator_def, Workspace* ws)
	      : Operator<CPUContext>(operator_def, ws) {
	      }
    void fillsrc(arm_compute::Tensor &tensor, const float* src, int N) {
      arm_compute::Window window;
      window.set(0, arm_compute::Window::Dimension(0, N, 1));
	std::cout << "finish src " << std::endl;
      
      arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates &id) {
        float value = src[id.x()];
	void *out = tensor.ptr_to_element(id);
	*reinterpret_cast<float *>(out) = value;
      });
    }
    void filldst(arm_compute::Tensor &tensor, float* dst, int N) {
      arm_compute::Window window;
      window.set(0, arm_compute::Window::Dimension(0, N, 1));
      
      arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates &id) {
	void *out = tensor.ptr_to_element(id);
        dst[id.x()] = *reinterpret_cast<float *>(out);
      });

    }

      bool RunOnDevice() override {
        auto& X = Input(0);
	auto* Y = Output(0);
	const int N = X.size();
	arm_compute::Tensor act0;
	arm_compute::Tensor src;
	arm_compute::TensorShape src_shape(N);
	arm_compute::NEActivationLayer act_lay;

	src.allocator()->init(arm_compute::TensorInfo(src_shape, 1, arm_compute::DataType::F32));
	act0.allocator()->init(arm_compute::TensorInfo(src_shape, 1, arm_compute::DataType::F32));
	act_lay.configure(&src, &act0, arm_compute::ActivationLayerInfo(arm_compute::ActivationLayerInfo::ActivationFunction::RELU));
	act0.allocator()->allocate();
	src.allocator()->allocate();
	const float *_src = X.template data<float>();
	fillsrc(src, _src, N); 

	act_lay.run();	

	float *_res = Y->template mutable_data<float>();
	filldst(act0, _res, N);

      }
  }; 
  REGISTER_CPU_OPERATOR_WITH_ENGINE(Relu, ARM, ARMReluOp);
}
