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

      unsigned int kernel_x_conv0 = filter.dim32(2);
      unsigned int kernel_y_conv0 = filter.dim32(3);
      unsigned int ofm_conv0 = M; 


      const arm_compute::TensorShape weight_shape(kernel_x_conv0, kernel_y_conv0, src_shape.z(), ofm_conv0);

      const arm_compute::TensorShape biases_shape(weight_shape[3]);
      const arm_compute::TensorShape out_shape(oW, oH, weight_shape[3]);

      weight.allocator()->init(arm_compute::TensorInfo(weight_shape, 1, arm_compute::DataType::F32));
      bias.allocator()->init(arm_compute::TensorInfo(biases_shape, 1, arm_compute::DataType::F32));
      conv.allocator()->init(arm_compute::TensorInfo(out_shape, 1, arm_compute::DataType::F32)); 

      conv.configure(&src, &weight, &bias, &conv, arm_compute::PadStrideInfo());
      src.allocator()->allocate();

      weight.allocator()->allocate();
      bias.allocator()->allocate();
      conv.allocator()->allocate();
      }
    bool convertinput(float* dst, const float* src, int N, int C, int H, int W) {
      for (unsigned int i = 0; i < N; i++) {
	for (unsigned int j = 0; j < C; j++) {
          for (unsigned int k = 0; k < H; k++) {
	    for (unsigned int h = 0; h < W; h++) {
              dst[h * H * C + k * C + j] = src[i * C * H * W + j * W * H + k * W + h];
	    }
	  }
        }
      }
      std::cout << "finish input converter " << std::endl;
    }
    bool convertweight(float* dst, const float* src, int out, int in, int kernel) {
      for (uint8_t i = 0; i < out; i++) {
        for (uint8_t j = 0; j < in; j++) {
          for (uint8_t k = 0; k < kernel; k++) {
            for (uint8_t h = 0; h < kernel; h++) {
	       dst[k * kernel * in * out + h * in * out + j * out + i] = src[i * in * kernel * kernel + j * kernel * kernel + k * kernel + h];
	    }
	  }
	}
      }
      std::cout << "finish weight converter " << std::endl;
    }
    bool convertres(float* src, float* dst, int N, int C, int H, int W) {
      for (unsigned int i = 0; i < N; i++) {
	for (unsigned int j = 0; j < C; j++) {
          for (unsigned int k = 0; k < H; k++) {
	    for (unsigned int h = 0; h < W; h++) {
              dst[i * C * H * W + j * W * H + k * W + h] = src[h * H * C + k * C + j];
	  }
	}
	}
      }
    }

    void fillsrc(arm_compute::Tensor &tensor, const float* src, int N, int C, int H, int W) {
      arm_compute::Window window;
      window.set(0, arm_compute::Window::Dimension(0, W, 1));
      window.set(1, arm_compute::Window::Dimension(0, H, 1));
      window.set(2, arm_compute::Window::Dimension(0, C, 1));
      
      arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates &id) {
        float value = src[id.z() * H * W + id.y() * W + id.x()];
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
        dst[id.z() * H * W + id.y() * W + id.x()] = *reinterpret_cast<float *>(out);
      });

    }
    bool RunOnDeviceWithOrderNCHW() override {
      auto& X = Input(0);
      auto& filter = Input(1);
      auto& bias = Input(2);
      auto* Y = Output(0);
 
      unsigned int N = X.dim32(0), C = X.dim32(1), H = X.dim32(2), W = X.dim32(3);
      unsigned int M = filter.dim32(0);  
      
      unsigned int oH = Y->dim32(2), oW = Y->dim32(3);

      unsigned int kernel_x_conv0 = filter.dim32(2);
      unsigned int kernel_y_conv0 = filter.dim32(3);
      unsigned int ofm_conv0 = M; 


      const float *_buf = X.template data<float>();
      float* buf = reinterpret_cast<float *>(src.allocator()->data());
      if (X.dim32(1) == src_shape.z() && X.dim32(2) == src_shape.y() && X.dim32(3) == src_shape.x()) {
        //convertinput(buf, _buf, N, C, H, W);
	fillsrc(src, _buf, N, C, H, W);
      }

      const float *_weights = filter.template data<float>();
      float* weights = reinterpret_cast<float *>(weights0.allocator()->data());
      if (filter.dim32(0) == ofm_conv0 && filter.dim32(1) == src_shape.z() && filter.dim32(2) == kernel_x_conv0 && filter.dim32(3) == kernel_x_conv0) {
	fillweights(weights0, _weights, ofm_conv0, src_shape.z(), kernel_x_conv0, kernel_x_conv0);
      }
      
      const float *_bias = bias.template data<float>();
      float *biass = reinterpret_cast<float *>(biases0.allocator()->data());
      for (unsigned int i = 0; i < ofm_conv0; i++) {
	      biass[i] = _bias[i];
      }

      conv0.run();
      
      float *_result = Y->template mutable_data<float>();
      if (Y->dim32(0) == 1 && Y->dim32(1) == weight_shape[3]) {
	std::cout << "dim correct " << std::endl;
        filldst(out_conv0, _result, 1, weight_shape[3], oH, oW);
      }
    }
  private:
      arm_compute::NEConvolutionLayer conv;
      arm_compute::Tensor src;
      arm_compute::Tensor weight;
      arm_compute::Tensor bias;
      arm_compute::Tensor res;
  };
  REGISTER_CPU_OPERATOR_WITH_ENGINE(Conv, ARM, ARMConvOp);
}
