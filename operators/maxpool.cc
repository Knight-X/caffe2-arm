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
  class ARMMaxPoolOp final : public ConvPoolOpBase<CPUContext> {
    public:
      ARMMaxPoolOp(const OperatorDef& operator_def, Workspace* ws) : ConvPoolOpBase<CPUContext>(operator_def, ws) 
      {
        auto& X = Input(0);
	auto* Y = Output(0);
	const unsigned int H = X.dim32(2), W = X.dim32(3);
	ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, X.dim32(1));
        const unsigned int oW = Y->dim32(3), oH = Y->dim32(2);
	const size_t N = X.dim32(0);
	const size_t C = X.dim32(1);

	const arm_compute::TensorShape input_shape(W, H, C);
	src.allocator()->init(arm_compute::TensorInfo(input_shape, 1, arm_compute::DataType::F32));
	arm_compute::TensorShape output_shape(oW, oH, C);

        res.allocator()->init(arm_compute::TensorInfo(output_shape, 1, arm_compute::DataType::F32));
	res.allocator()->allocate();
	src.allocator()->allocate();
		
        pool.configure(&src, &res, arm_compute::PoolingLayerInfo(arm_compute::PoolingType::MAX, 2));
      }

    void fillsrc(arm_compute::Tensor &tensor, const float* src, int N, int C, int H, int W) {
      arm_compute::Window window;
      window.set(0, arm_compute::Window::Dimension(0, W, 1));
      window.set(1, arm_compute::Window::Dimension(0, H, 1));
      window.set(2, arm_compute::Window::Dimension(0, C, 1));
      
      arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates &id) {
        float value = src[N * C * H * W + id.z() * H * W + id.y() * W + id.x()];
	void *out = tensor.ptr_to_element(id);
	*reinterpret_cast<float *>(out) = value;
      });

    }
    void filldst(arm_compute::Tensor &tensor, float* dst, int N, int C, int H, int W) {
      arm_compute::Window window;
      window.set(0, arm_compute::Window::Dimension(0, W, 1));
      window.set(1, arm_compute::Window::Dimension(0, H, 1));
      window.set(2, arm_compute::Window::Dimension(0, C, 1));
      
      arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates &id) {
	void *out = tensor.ptr_to_element(id);
        dst[N * C * H * W + id.z() * H * W + id.y() * W + id.x()] = *reinterpret_cast<float *>(out);
      });

    }

      bool RunOnDeviceWithOrderNCHW() override {
        auto& X = Input(0);
	auto* Y = Output(0);
	const unsigned int H = X.dim32(2), W = X.dim32(3);
        const unsigned int oW = Y->dim32(3), oH = Y->dim32(2);
	const size_t N = X.dim32(0);
	const size_t C = X.dim32(1);

	
	for (unsigned int n = 0; n < N; n++) {
	  const float *_src = X.template data<float>();
	  fillsrc(src, _src, n, C, H, W);
	  pool.run();
	  float *_res = Y->template mutable_data<float>();
	  filldst(res, _res, n, C, oH, oW);
	}
	return true;

      }
    private:
	arm_compute::NEPoolingLayer pool;
	arm_compute::Tensor res;
	arm_compute::Tensor src;
    };
REGISTER_CPU_OPERATOR_WITH_ENGINE(MaxPool, ARM, ARMMaxPoolOp);
}


