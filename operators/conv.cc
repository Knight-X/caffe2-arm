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
  class ARMConvOp final : public ConvPoolOpBase<CPUContext> {
  public:
    ARMConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CPUContext>(operator_def, ws) {
      auto& X = Input(0);
      auto& filter = Input(1);
      auto& bias = Input(2);
      auto* Y = Output(0);
 
      unsigned int N = X.dim32(0), C = X.dim32(1), H = X.dim32(2), W = X.dim32(3);
      unsigned int M = filter.dim32(0);  
      
      const arm_compute::TensorShape src_shape(W, H, C);
      src.allocator()->init(arm_compute::TensorInfo(src_shape, 1, arm_compute::DataType::F32));
      
      ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, filter.dim32(0));
      unsigned int oH = Y->dim32(2), oW = Y->dim32(3);

      unsigned int kernel_x_conv = filter.dim32(2);
      unsigned int kernel_y_conv = filter.dim32(3);
      unsigned int ofm = M; 


      const arm_compute::TensorShape weight_shape(kernel_x_conv, kernel_y_conv, src_shape.z(), ofm);

      const arm_compute::TensorShape biases_shape(weight_shape[3]);
      const arm_compute::TensorShape out_shape(oW, oH, weight_shape[3]);

      weightes.allocator()->init(arm_compute::TensorInfo(weight_shape, 1, arm_compute::DataType::F32));
      biases.allocator()->init(arm_compute::TensorInfo(biases_shape, 1, arm_compute::DataType::F32));
      res.allocator()->init(arm_compute::TensorInfo(out_shape, 1, arm_compute::DataType::F32)); 

      conv.configure(&src, &weightes, &biases, &res, arm_compute::PadStrideInfo());
      src.allocator()->allocate();

      weightes.allocator()->allocate();
      biases.allocator()->allocate();
      res.allocator()->allocate();
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
    void fillweights(arm_compute::Tensor &tensor, const float* src, int out, int in, int x, int y) {
      arm_compute::Window window;
      window.set(0, arm_compute::Window::Dimension(0, x, 1));
      window.set(1, arm_compute::Window::Dimension(0, y, 1));
      window.set(2, arm_compute::Window::Dimension(0, in, 1));
      window.set(3, arm_compute::Window::Dimension(0, out, 1));
      
      arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates &id) {
        float value = src[id[3] * in * x * y + id[2] * x * y + id[1] * x + id[0]];
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
    void fillbias(arm_compute::Tensor &tensor, const float* src, int N) {
      arm_compute::Window window;
      window.set(0, arm_compute::Window::Dimension(0, N, 1));
      
      arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates &id) {
	float value = src[id.x()];
	void *out = tensor.ptr_to_element(id);
        *reinterpret_cast<float *>(out) = value;
      });

    }
    bool RunOnDeviceWithOrderNCHW() override {
      auto& X = Input(0);
      auto& filter = Input(1);
      auto& bias = Input(2);
      auto* Y = Output(0);
 
      unsigned int N = X.dim32(0), C = X.dim32(1), H = X.dim32(2), W = X.dim32(3);
      unsigned int M = filter.dim32(0);  
      
      const arm_compute::TensorShape src_shape(W, H, C);
      unsigned int oH = Y->dim32(2), oW = Y->dim32(3);

      unsigned int kernel_x_conv = filter.dim32(2);
      unsigned int kernel_y_conv = filter.dim32(3);
      unsigned int ofm = M; 
      const arm_compute::TensorShape weight_shape(kernel_x_conv, kernel_y_conv, src_shape.z(), ofm);
      const float *_weight = filter.template data<float>();
      if (filter.dim32(0) == ofm && filter.dim32(1) == src_shape.z() && filter.dim32(2) == kernel_x_conv && filter.dim32(3) == kernel_x_conv) {
	fillweights(weightes, _weight, ofm, src_shape.z(), kernel_x_conv, kernel_x_conv);
      }

      const float *_bias = bias.template data<float>();
      fillbias(biases, _bias, ofm); 

      for (unsigned int n = 0; n < N; n++) {

      	const float *_buf = X.template data<float>();
      	if (X.dim32(1) == src_shape.z() && X.dim32(2) == src_shape.y() && X.dim32(3) == src_shape.x()) {
	  fillsrc(src, _buf, n, C, H, W);
        }
      

        conv.run();
      
        float *_res = Y->template mutable_data<float>();
        if (Y->dim32(0) == 1 && Y->dim32(1) == weight_shape[3]) {
	 std::cout << "dim correct " << std::endl;
          filldst(res, _res, n, weight_shape[3], oH, oW);
        }
      }
      return true;
    }
  private:
      arm_compute::NEConvolutionLayer conv;
      arm_compute::Tensor src;
      arm_compute::Tensor weightes;
      arm_compute::Tensor biases;
      arm_compute::Tensor res;
  };
  REGISTER_CPU_OPERATOR_WITH_ENGINE(Conv, ARM, ARMConvOp);
}
