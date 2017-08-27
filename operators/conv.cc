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
      : ConvPoolOpBase<CPUContext>(operator_def, ws) {}
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
      std::cout << "HELLO I AM ARM ENgine" << std::endl;
      auto& X = Input(0);
      auto& filter = Input(1);
      auto& bias = Input(2);
      auto* Y = Output(0);
 
      unsigned int N = X.dim32(0), C = X.dim32(1), H = X.dim32(2), W = X.dim32(3);
      //unsigned int N = 2, H = 32, W = 32;
      unsigned int M = filter.dim32(0);  
      arm_compute::NEConvolutionLayer conv0;
      arm_compute::Tensor src;
      arm_compute::Tensor weights0;
      arm_compute::Tensor biases0;
      arm_compute::Tensor out_conv0;
      
      const arm_compute::TensorShape src_shape(W, H, C);
      src.allocator()->init(arm_compute::TensorInfo(src_shape, 1, arm_compute::DataType::F32));
      
      ConvPoolOpBase<CPUContext>::SetOutputSize(X, Y, filter.dim32(0));
      unsigned int oH = Y->dim32(2), oW = Y->dim32(3);
      //unsigned int oH = 28, oW = 28;

      unsigned int kernel_x_conv0 = filter.dim32(2);
      unsigned int kernel_y_conv0 = filter.dim32(3);
      unsigned int ofm_conv0 = M; 
      //unsigned int kernel_x_conv0 = 5;
      //unsigned int kernel_y_conv0 = 5;
      //unsigned int ofm_conv0 = 8;
      const arm_compute::TensorShape weight_shape(kernel_x_conv0, kernel_y_conv0, src_shape.z(), ofm_conv0);

      const arm_compute::TensorShape biases_shape(weight_shape[3]);
      const arm_compute::TensorShape out_shape(oW, oH, weight_shape[3]);
      weights0.allocator()->init(arm_compute::TensorInfo(weight_shape, 1, arm_compute::DataType::F32));
      biases0.allocator()->init(arm_compute::TensorInfo(biases_shape, 1, arm_compute::DataType::F32));
      out_conv0.allocator()->init(arm_compute::TensorInfo(out_shape, 1, arm_compute::DataType::F32)); 

      conv0.configure(&src, &weights0, &biases0, &out_conv0, arm_compute::PadStrideInfo());
      src.allocator()->allocate();

      weights0.allocator()->allocate();
      biases0.allocator()->allocate();
      out_conv0.allocator()->allocate();


      std::cout << "start get x data" << std::endl;
      const float *_buf = X.template data<float>();
      float* buf = reinterpret_cast<float *>(src.allocator()->data());
      if (X.dim32(1) == src_shape.z() && X.dim32(2) == src_shape.y() && X.dim32(3) == src_shape.x()) {
        //convertinput(buf, _buf, N, C, H, W);
	fillsrc(src, _buf, N, C, H, W);
      }
      /*for (unsigned int i = 0; i < N; i++) {
        for (unsigned int j = 0; j < C; j++) {
	  for (unsigned int k = 0; k < H; k++) {
	    for (unsigned int h = 0; h < W; h++) {
	      std::cout << _buf[i * C * H * W + j * H * W + k * W + h] << " ";
  	    }
  	  }
	}
      }*/	
      /*
      uint8_t *tmp = buf.data();
      uint8_t *rt = new uint8_t(X.size());
      convertres(tmp, rt, N, C, H, W);
      for (unsigned int i = 0; i < X.size(); i++) {
        std::cout << unsigned(rt[i]) << " ";
      }
      */
      std::cout << "end get x data size is : " << X.size() << std::endl;
      std::cout << "start get weight data" << std::endl;

      const float *_weights = filter.template data<float>();
      float* weights = reinterpret_cast<float *>(weights0.allocator()->data());
      std::cout << filter.dim32(0) << " " << filter.dim32(1) << " " << X.dim32(0) << " " << filter.dim32(2) << " " << filter.dim32(3) << std::endl;
      std::cout << ofm_conv0 << " " << src_shape.z() << " " << kernel_x_conv0 << " " << std::endl;
      if (filter.dim32(0) == ofm_conv0 && filter.dim32(1) == src_shape.z() && filter.dim32(2) == kernel_x_conv0 && filter.dim32(3) == kernel_x_conv0) {
        //convertweight(weights, _weights, ofm_conv0, src_shape.z(), kernel_x_conv0);
	fillweights(weights0, _weights, ofm_conv0, src_shape.z(), kernel_x_conv0, kernel_x_conv0);
      }
      
      std::cout << "start get biases data" << std::endl;
      const float *_bias = bias.template data<float>();
      float *biass = reinterpret_cast<float *>(biases0.allocator()->data());
      for (unsigned int i = 0; i < ofm_conv0; i++) {
	      biass[i] = _bias[i];
      }
      std::cout << "end get biases data" << std::endl;

      /*uint8_t *_resa = out_conv0.allocator()->data();
      for (unsigned int i = 0; i < 1; i++) {
        for (unsigned int j = 0; j < weight_shape[3]; j++) {
	  for (unsigned int x = 0; x < oH; x++) {
	    for (unsigned int y = 0; y < oW; y++) {
	      std::cout << unsigned(_resa[y * oH * weight_shape[3] + x * weight_shape[3] + j + i]) << " ";
	    }
	  }
	  std::cout << "one channel" << std::endl;
	}
	  std::cout << std::endl;
      }*/
      
     
      conv0.run();
      
      /*float *rt = reinterpret_cast<float *>(weights0.allocator()->data());
      for (unsigned int i = 0; i < ofm_conv0; i++) {
        for (unsigned int j = 0; j < src_shape[3]; j++) {
	  for (unsigned int x = 0; x < kernel_x_conv0; x++) {
	    for (unsigned int y = 0; y < kernel_x_conv0; y++) {
	      std::cout << rt[x * kernel_x_conv0 * ofm_conv0 * src_shape[3] + y * ofm_conv0 * src_shape[3] + j * ofm_conv0 + i] << " ";
	    }
	  }
	  std::cout << std::endl;
	}
      }
      float *biast = reinterpret_cast<float *>(biases0.allocator()->data());
      for (unsigned int i = 0; i < ofm_conv0; i++) {
        std::cout << biast[i] << " ";
      }
      float* _resa = reinterpret_cast<float*>(out_conv0.allocator()->data());
      for (unsigned int i = 0; i < oW; i++) {
        for (unsigned int j = 0; j < oH; j++) {
	  for (unsigned int x = 0; x < weight_shape[3]; x++) {
	      std::cout << _resa[i * oH * weight_shape[3] + j * weight_shape[3] + x] << " ";
	    }
	  std::cout << "one channel" << std::endl;
	}
      }*/
      
      float *_result = Y->template mutable_data<float>();
      if (Y->dim32(0) == 1 && Y->dim32(1) == weight_shape[3]) {
	std::cout << "dim correct " << std::endl;
        filldst(out_conv0, _result, 1, weight_shape[3], oH, oW);
      }
    }
  private:
  };
  REGISTER_CPU_OPERATOR_WITH_ENGINE(Conv, ARM, ARMConvOp);
}
