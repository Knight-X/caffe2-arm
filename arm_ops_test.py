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


dyndep.InitOpsLibrary("./conv.so")


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

if __name__ == "__main__":
    import unittest
    unittest.main()

