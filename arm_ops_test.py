from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import hypothesis.strategies as st
from hypothesis import given, assume, settings
import numpy as np
import time
import os
from caffe2.python import core, dyndep
import caffe2.python.hypothesis_test_util as hu


dyndep.InitOpsLibrary("./arm.so")


def benchmark(ws, net, warmups=5, iters=100):
    for _ in range(warmups):
        ws.run(net)
    plan = core.Plan("plan")
    plan.AddStep(core.ExecutionStep("test-step", net, iters))
    before = time.time()
    ws.run(plan)
    after = time.time()
    print("Timing network, time taken per-iteration: {:.6f}ms".format((
        after - before) / float(iters) * 1000.0))
    return after - before


def has_avx2():
    import subprocess
    try:
        subprocess.check_output(["grep", "avx2", "/proc/cpuinfo"])
        return True
    except subprocess.CalledProcessError:
        # grep exits with rc 1 on no matches
        return False


class NNPackOpsTest(hu.HypothesisTestCase):
    @given(stride=st.integers(1, 1),
           pad=st.integers(0, 0),
           kernel=st.integers(3, 5),
           size=st.integers(5 , 32),
           input_channels=st.integers(2, 2),
           output_channels=st.integers(2, 2),
           batch_size=st.integers(1, 1),
           groups=st.integers(1, 1))
    def test_convolution_correctness(self, stride, pad, kernel, size,
                                     input_channels, output_channels,
                                     batch_size, groups):
        assume(input_channels % groups == 0)
        assume(output_channels % groups == 0)
        #assume(output_channels == input_channels / groups)
        assume(stride <= kernel)
        if stride != 1:
            assume(batch_size == 1)

        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32)
        for i in range(size):
            print(X[0][0][i])
        w = np.random.rand(
            output_channels, input_channels, kernel, kernel).astype(np.float32)
        b = np.random.rand(output_channels).astype(np.float32)
        order = "NCHW"
        outputs = {}
        for engine in ["", "ARM"]:
            op = core.CreateOperator(
                "Conv",
                ["X", "w", "b"],
                ["Y"],
                stride=stride,
                kernel=kernel,
                pad=pad,
                order=order,
                kts="TUPLE",
                engine=engine,
                group=groups,
            )
            self.ws.create_blob("X").feed(X)
            self.ws.create_blob("w").feed(w)
            self.ws.create_blob("b").feed(b)
            self.ws.run(op)
            outputs[engine] = self.ws.blobs["Y"].fetch()
        np.testing.assert_allclose(
            outputs[""],
            outputs["ARM"],
            atol=1e-4,
            rtol=1e-4)

    @given(size=st.sampled_from([6, 8]),
           input_channels=st.integers(1, 8),
           batch_size=st.integers(1, 1))
    def test_max_pool_correctness(self, size, input_channels, batch_size):
        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32) - 0.5
        order = "NCHW"
        outputs = {}
        # only 2 * 2 stride and 2 * 2 pool is supported in NNPack now
        stride = 1 
        kernel = 2
        # The pooling strategy of NNPack is different from caffe2 pooling
        pad = 0
        for engine in ["", "ARM"]:
            op = core.CreateOperator(
                "MaxPool",
                ["X"],
                ["Y"],
                stride=stride,
                kernel=kernel,
                pad=pad,
                order=order,
                engine=engine,
            )
            self.ws.create_blob("X").feed(X)
            self.ws.run(op)
            outputs[engine] = self.ws.blobs["Y"].fetch()
        np.testing.assert_allclose(
            outputs[""],
            outputs["ARM"],
            atol=1e-4,
            rtol=1e-4)

    @given(size=st.sampled_from([6, 8]),
           input_channels=st.integers(1, 8),
           batch_size=st.integers(1, 5))
    def test_relu_correctness(self, size, input_channels, batch_size):
        X = np.random.rand(
            batch_size, input_channels, size, size).astype(np.float32) - 0.5
        outputs = {}
        for engine in ["", "ARM"]:
            op = core.CreateOperator(
                "Relu",
                ["X"],
                ["Y"],
                engine=engine,
            )
            self.ws.create_blob("X").feed(X)
            self.ws.run(op)
            outputs[engine] = self.ws.blobs["Y"].fetch()
        np.testing.assert_allclose(
            outputs[""],
            outputs["ARM"],
            atol=1e-4,
            rtol=1e-4)

    @given(size=st.sampled_from([6, 8]),
           input_channels=st.integers(1, 8),
           batch_size=st.integers(1, 1))
    def test_softmax_correctness(self, size, input_channels, batch_size):
        X = np.random.rand(
            batch_size, size).astype(np.float32) - 0.5
        outputs = {}
        for engine in ["", "ARM"]:
            op = core.CreateOperator(
                "Softmax",
                ["X"],
                ["Y"],
                engine=engine,
            )
            self.ws.create_blob("X").feed(X)
            self.ws.run(op)
            outputs[engine] = self.ws.blobs["Y"].fetch()
        np.testing.assert_allclose(
            outputs[""],
            outputs["ARM"],
            atol=1e-4,
            rtol=1e-4
            )

    @given(n=st.integers(1, 2), m=st.integers(1, 1),
           k=st.integers(3, 5),
           multi_dim=st.sampled_from([False]),
           **hu.gcs)
    def test_fc(self, n, m, k, multi_dim, gc, dc):
        X = np.random.rand(m, k).astype(np.float32) - 0.5
        W = np.random.rand(n, k).astype(np.float32) - 0.5
        b = np.random.rand(n).astype(np.float32) - 0.5
        outputs = {}
        
        for engine in ["", "ARM"]:
            op = core.CreateOperator(
                "FC",
                ["X", "W", "b"],
                ["Y"],
                engine=engine,
            )
            self.ws.create_blob("X").feed(X)
            self.ws.create_blob("W").feed(W)
            self.ws.create_blob("b").feed(b)
            self.ws.run(op)
            outputs[engine] = self.ws.blobs["Y"].fetch()
        np.testing.assert_allclose(
            outputs[""],
            outputs["ARM"],
            atol=1e-4,
            rtol=1e-4
            )


#        def fc_op(X, W, b):
#            return [np.dot(X, W.reshape(n, k).transpose()) + b.reshape(n)]

 #       op = core.CreateOperator(
  #          'FC',
   #         ['X', 'W', 'b'],
    #        'out'
    #    )

        # Check against numpy reference
     #   np.testing.assert_allclose(
     #       device_option=gc,
     #       op=op,
     #       inputs=[X, W, b],
     #       reference=fc_op,
     #   )

if __name__ == "__main__":
    import unittest
    unittest.main()

