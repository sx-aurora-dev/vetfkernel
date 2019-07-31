"""Functional tests for pooling operations."""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test

class PoolingTest(test.TestCase):
  @test_util.run_deprecated_v1
  def testAvgPoolGrad(self):
    self._testAvgPoolGradSamePadding4('NCHW')

  def _testAvgPoolGradSamePadding4(self, data_format):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[1, 3, 3, 1],
        output_sizes=[1, 3, 3, 1],
        window_rows=3,
        window_cols=3,
        row_stride=1,
        col_stride=1,
        padding="SAME",
        data_format=data_format,
        use_gpu=True)

  def _ConstructAndTestGradient(self,
                                pool_func,
                                input_sizes,
                                output_sizes,
                                window_rows,
                                window_cols,
                                row_stride,
                                col_stride,
                                padding,
                                data_format,
                                use_gpu,
                                x_init_value=None):
    """Verifies the gradients of the avg pooling function.

    Args:
      pool_func: Function to be called, co.MaxPool, co.AvgPool,
        or the Lua version.
      input_sizes: Input tensor dimensions.
      output_sizes: Output tensor dimensions.
      window_rows: kernel size in row dim
      window_cols: kernel size in col dim
      row_stride: Row Stride.
      col_stride: Col Stride.
      padding: Padding type.
      data_format: Data format.
      use_gpu: whether we are running on GPU
      x_init_value: Values to be passed to the gradient checker.
    """
    assert input_sizes[0] == output_sizes[0]
    assert input_sizes[3] == output_sizes[3]
    total_size = 1
    for s in input_sizes:
      total_size *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x = [f * 1.0 for f in range(1, total_size + 1)]
    with self.cached_session(use_gpu=use_gpu):
      input_tensor = constant_op.constant(x, shape=input_sizes, name="input")
      func_name = "avg_pool"
      err_tolerance = 1e-4
      if data_format == "NCHW":
        ksize = [1, 1, window_rows, window_rows]
        strides = [1, 1, row_stride, col_stride]
        t = test_util.NHWCToNCHW(input_tensor)
      else:
        ksize = [1, window_rows, window_rows, 1]
        strides = [1, row_stride, col_stride, 1]
        t = input_tensor
      t = pool_func(
          t,
          ksize=ksize,
          strides=strides,
          padding=padding,
          data_format=data_format,
          name=func_name)
      if data_format == "NCHW":
        t = test_util.NCHWToNHWC(t)

      err = gradient_checker.compute_gradient_error(
          input_tensor,
          input_sizes,
          t,
          output_sizes,
          x_init_value=x_init_value,
          delta=1e-2)
#    tf_logging.info("%s gradient error = " % func_name, err)
    self.assertLess(err, err_tolerance)


if __name__ == "__main__":
  test.main()
