
import numpy as np
import math

import mindspore as ms
from mindspore._checkparam import twice

class _Conv(ms.nn.Cell):
    r"""Applies a N-D convolution over an input signal composed of several input
       planes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 pad_mode,
                 padding,
                 dilation,
                 group,
                 data_format,
                 has_bias,
                 weight_init,
                 bias_init,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_mode = pad_mode
        self.padding = padding
        self.dilation = dilation
        self.group = group
        self.data_format = data_format
        self.has_bias = has_bias
        if not (isinstance(in_channels, int) and in_channels > 0):
            raise ValueError('Attr \'in_channels\' of \'Conv2D\' Op passed '
                             + str(in_channels) + ', should be a int and greater than 0.')
        if (not isinstance(kernel_size, tuple)) or len(kernel_size) != 2 or \
                (not isinstance(kernel_size[0], int)) or (not isinstance(kernel_size[1], int)) or \
                kernel_size[0] < 1 or kernel_size[1] < 1:
            raise ValueError('Attr \'kernel_size\' of \'Conv2D\' Op passed '
                             + str(self.kernel_size) + ', should be a int or tuple and equal to or greater than 1.')
        if in_channels % group != 0:
            raise ValueError('Attr \'in_channels\' of \'Conv2D\' Op must be divisible by '
                             'attr \'group\' of \'Conv2D\' Op.')
        if out_channels % group != 0:
            raise ValueError('Attr \'out_channels\' of \'Conv2D\' Op must be divisible by '
                             'attr \'group\' of \'Conv2D\' Op.')

        self.weight = ms.Parameter(ms.common.initializer.initializer(
            weight_init, [out_channels, in_channels // group, *kernel_size]), name='weight')

        if has_bias:
            self.bias = ms.Parameter(_initializer(
                bias_init, [out_channels]), name='bias')
        else:
            if bias_init != 'zeros':
                logger.warning("Value of 'has_bias' is False, value of 'bias_init' will be ignored.")
            self.bias = None

    def construct(self, *inputs):
        raise NotImplementedError


class Conv2d_SENG_GPU(_Conv):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_mode='same',
                 padding=0,
                 dilation=1,
                 group=1,
                 data_format='NCHW',
                 has_bias=False,
                 weight_init='normal',
                 damping=0.03,
                 bias_init='zeros',
                 input_hw=(8, 8),
                 extra_args=None):
        self.seng = extra_args.is_train
        self.hw = kernel_size * kernel_size

        batch_size = extra_args.batch_size
        frequency = extra_args.frequency
        loss_scale = extra_args.loss_scale
        col_sample_size = extra_args.col_sample_size
        im_size_threshold = extra_args.im_size_threshold

        if has_bias:
            self.bias_shape = (out_channels,)
        kernel_size = twice(kernel_size)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            pad_mode,
            padding,
            dilation,
            group,
            data_format,
            has_bias,
            weight_init,
            bias_init,
        )

        self.batch_size = batch_size
        self.batched_input_shape = (batch_size, in_channels, *input_hw)
        self.weight_shape = (out_channels, in_channels // group, *kernel_size)
        self.col_sample_size = col_sample_size
        self.bias_shape = None

        self.conv2d = ms.ops.Conv2D(out_channel=self.out_channels,
                               kernel_size=self.kernel_size,
                               mode=1,
                               pad_mode=self.pad_mode,
                               pad=self.padding,
                               stride=self.stride,
                               dilation=self.dilation,
                               group=self.group
                               )

        # self.matrix_A_dim = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.img2col = ms.ops.operations.Im2Col(kernel_size=kernel_size, stride=stride, pad_mode=pad_mode)

        img2col_shape = self.img2col.infer_shape(self.batched_input_shape)
        matrix_A_shape = (batch_size, img2col_shape[0] * img2col_shape[1] * img2col_shape[2], img2col_shape[4]*img2col_shape[5])
        output_shape = self.infer_shape()
        matrix_G_shape = (batch_size, output_shape[1], output_shape[2] * output_shape[3])
        parameter_numel = matrix_G_shape[1] * matrix_A_shape[1]
        self.matrix_U_shape = (batch_size, parameter_numel)
        self.freq = ms.Tensor(frequency, ms.int32)

        self.input_tensor = ms.Parameter(np.zeros(self.batched_input_shape).astype(np.float32), name='input_tensor', requires_grad=False)
        self.matrix_iUUt = ms.Parameter(np.zeros((batch_size, batch_size)).astype(np.float32), name='matrix_iUUt', requires_grad=False)
        self.cov_step = ms.Parameter(ms.common.initializer.initializer(0, [1], ms.int32), name="cov_step", requires_grad=False)
        self.fake_current_damping = ms.Parameter(np.zeros((1,)).astype(np.float32), name="fake_current_damping", requires_grad=False)

        is_implicit_representation = (im_size_threshold < parameter_numel)
        self.is_implicit_representation = is_implicit_representation
        if self.is_implicit_representation:
            self.is_sample_A = col_sample_size < matrix_A_shape[1]
            self.is_sample_G = col_sample_size < matrix_G_shape[1]
            self.is_col_sample = self.is_sample_A or self.is_sample_G

            self.matrix_A = ms.Parameter(np.zeros(matrix_A_shape).astype(np.float32), name='matrix_A', requires_grad=False)
            self.matrix_G = ms.Parameter(np.zeros(matrix_G_shape).astype(np.float32), name='matrix_G', requires_grad=False)
            if self.is_col_sample:
                self.matrix_U_all_index = ms.Tensor(np.arange(parameter_numel, dtype=np.float16).reshape(matrix_G_shape[1], matrix_A_shape[1]))

                self.matrix_sub_G_shape = (matrix_G_shape[0], min(matrix_G_shape[1], col_sample_size), matrix_G_shape[2])
                self.matrix_sub_A_shape = (matrix_A_shape[0], min(matrix_A_shape[1], col_sample_size), matrix_A_shape[2])

                real_col_sample_size = self.matrix_sub_G_shape[1] * self.matrix_sub_A_shape[1]

                self.matrix_sub_A = ms.Parameter(np.zeros(self.matrix_sub_A_shape).astype(np.float32), name='matrix_sub_A', requires_grad=False)
                self.matrix_sub_G = ms.Parameter(np.zeros(self.matrix_sub_G_shape).astype(np.float32), name='matrix_sub_G', requires_grad=False)
                self.sample_index = ms.Parameter(np.zeros(real_col_sample_size).astype(np.int32), name='sample_index', requires_grad=False)
            else:
                # fake param, ensuring that every parameter is defined in every execution path
                self.matrix_sub_A = ms.Parameter(np.zeros((1, 1, 1)).astype(np.float32), name='matrix_sub_A', requires_grad=False)
                self.matrix_sub_G = ms.Parameter(np.zeros((1, 1, 1)).astype(np.float32), name='matrix_sub_G', requires_grad=False)
                self.sample_index = ms.Parameter(np.zeros((1,)).astype(np.int32), name='sample_index', requires_grad=False)

            if self.is_sample_A:
                self.rnd_choice_mask_A = ms.ops.RandomChoiceWithMask(count=col_sample_size)
                self.cols_as_mask_A = ms.Tensor(np.ones(shape=[self.matrix_sub_A_shape[1]]).astype(np.bool))
                self.mul_ratio_A = ms.Tensor(self.matrix_sub_A_shape[1] / col_sample_size, ms.float32)
            else:
                self.sample_index_A = ms.Tensor(np.arange(matrix_A_shape[1], dtype=np.int32))

            if self.is_sample_G:
                self.rnd_choice_mask_G = ms.ops.RandomChoiceWithMask(count=col_sample_size)
                self.cols_as_mask_G = ms.Tensor(np.ones(shape=[self.matrix_sub_G_shape[1]]).astype(np.bool))
                self.mul_ratio_G = ms.Tensor(self.matrix_sub_G_shape[1] / col_sample_size, ms.float32)
            else:
                self.sample_index_G = ms.Tensor(np.arange(matrix_G_shape[1], dtype=np.int32))
            # fake param
            self.matrix_sub_U = ms.Parameter(np.zeros((1, 1)).astype(np.float32), name='matrix_sub_U', requires_grad=False)
            self.matrix_U = ms.Parameter(np.zeros((1, 1)).astype(np.float32), name='matrix_U', requires_grad=False)
        else:
            # explicit U
            self.matrix_U = ms.Parameter(np.zeros(self.matrix_U_shape).astype(np.float32), name='matrix_U', requires_grad=False)
            real_col_sample_size = col_sample_size * col_sample_size
            self.is_col_sample = real_col_sample_size < parameter_numel

            if self.is_col_sample:
                self.matrix_sub_U_shape = (batch_size, real_col_sample_size)
                self.matrix_sub_U = ms.Parameter(np.zeros(self.matrix_sub_U_shape).astype(np.float32), name='matrix_sub_U', requires_grad=False)
                self.sample_index = ms.Parameter(np.zeros(real_col_sample_size).astype(np.int32), name='sample_index', requires_grad=False)
                self.rnd_choice_mask = ms.ops.RandomChoiceWithMask(count=real_col_sample_size)
                self.cols_as_mask = ms.Tensor(np.ones(shape=[parameter_numel]).astype(np.bool))

                self.mul_ratio = ms.Tensor(parameter_numel / real_col_sample_size, ms.float32)
            else:
                # fake param
                self.matrix_sub_U = ms.Parameter(np.zeros((1, 1)).astype(np.float32), name='matrix_sub_U', requires_grad=False)
                self.sample_index = ms.Parameter(np.zeros(1).astype(np.int32), name='sample_index', requires_grad=False)
            # fake param
            self.matrix_A = ms.Parameter(np.zeros((1, 1, 1)).astype(np.float32), name='matrix_A', requires_grad=False)
            self.matrix_G = ms.Parameter(np.zeros((1, 1, 1)).astype(np.float32), name='matrix_G', requires_grad=False)
            self.matrix_sub_A = ms.Parameter(np.zeros((1, 1, 1)).astype(np.float32), name='matrix_sub_A', requires_grad=False)
            self.matrix_sub_G = ms.Parameter(np.zeros((1, 1, 1)).astype(np.float32), name='matrix_sub_G', requires_grad=False)

        # For train.py to test layer type
        layer_seng_type = 1
        if self.is_col_sample:
            layer_seng_type += 2
        if is_implicit_representation:
            layer_seng_type += 4
        self.layer_seng_type = ms.Parameter(np.zeros(layer_seng_type).astype(np.bool), name='layer_seng_type', requires_grad=False)

        self.matmul_at = ms.ops.MatMul(transpose_a=True)
        self.matmul_bt = ms.ops.MatMul(transpose_b=True)
        self.matrix_G_normalizer = ms.Tensor(batch_size**0.5 / loss_scale, ms.float32)
        self.transpose = ms.ops.Transpose()
        self.bmm_bt = ms.ops.BatchMatMul(transpose_b=True)
        self.reduce_sum = ms.ops.ReduceSum(keep_dims=False)
        self.damping = ms.Parameter(ms.Tensor(damping, dtype=ms.float32), name="damping_value", requires_grad=False)
        self.ops_eye = ms.ops.Eye()
        self.gather = ms.ops.Gather()
        self.getG = ms.ops.InsertGradientOf(self.save_gradient)
        self.cholesky = ms.ops.operations.CholeskyTrsm()

    def save_gradient(self, dout):
        matrix_A = self.img2col(self.input_tensor) #(in_channels, kernel_h, kernel_w, batch_size, out_h, out_w)
        tmp0 = self.in_channels*self.kernel_size[0]*self.kernel_size[1], self.batch_size, -1
        matrix_A = ms.ops.cast(self.transpose(ms.ops.reshape(matrix_A, tmp0), (1, 0, 2)), ms.float32)
        # (batch_size, in_channels*kernel_h*kernel_w, out_h*out_w)

        matrix_G = self.matrix_G_normalizer * ms.ops.cast(dout, ms.float32) #(batch_size, out_channel, out_h, out_w)
        matrix_G = ms.ops.reshape(matrix_G, (self.batch_size,self.out_channels,-1))
        # (batch_size,out_channels,out_h*out_w)

        if self.is_implicit_representation:
            self.matrix_A = matrix_A
            self.matrix_G = matrix_G
            if self.is_col_sample:
                if self.is_sample_A:
                    sample_index_A = self.rnd_choice_mask_A(self.cols_as_mask_A)[0][:,0]
                    matrix_sub_A = self.mul_ratio_A * self.gather(matrix_A, sample_index_A, 1)
                    sample_index = self.gather(self.matrix_U_all_index, sample_index_A, 1)
                else:
                    sample_index = self.matrix_U_all_index
                    sample_index_A = self.sample_index_A
                    matrix_sub_A = matrix_A

                if self.is_sample_G:
                    sample_index_G = self.rnd_choice_mask_G(self.cols_as_mask_G)[0][:,0]
                    matrix_sub_G = self.mul_ratio_G * self.gather(matrix_G, sample_index_G, 1)
                    sample_index = self.gather(sample_index, sample_index_G, 0)
                else:
                    sample_index_G = self.sample_index_G
                    matrix_sub_G = matrix_G

                sample_index = ms.ops.cast(ms.ops.reshape(sample_index, (-1,)), ms.int32)
                self.sample_index = sample_index
                self.matrix_sub_A = matrix_sub_A
                self.matrix_sub_G = matrix_sub_G
            else:
                matrix_sub_A = matrix_A
                matrix_sub_G = matrix_G

            # matrix_UUt = np.einsum(matA,[0,1,2],matA,[3,1,4],matG,[0,5,2],matG,[3,5,4],[0,3]) # equivalent numpy code
            tmp0 = ms.ops.reshape(self.bmm_bt(matrix_sub_A, matrix_sub_G), (self.batch_size,-1))
            matrix_UUt = self.matmul_bt(tmp0, tmp0)

        else:
            matrix_U = ms.ops.reshape(self.bmm_bt(matrix_G, matrix_A), (self.batch_size, -1))
            self.matrix_U = matrix_U
            if self.is_col_sample:
                sample_index = self.rnd_choice_mask(self.cols_as_mask)[0][:,0]
                matrix_sub_U = self.mul_ratio * self.gather(matrix_U, sample_index, 1)
                self.sample_index = sample_index
                self.matrix_sub_U = matrix_sub_U
            else:
                matrix_sub_U = matrix_U
            matrix_UUt = self.matmul_bt(matrix_sub_U, matrix_sub_U)

        self.cov_step = self.cov_step + self.freq
        damped_UUt = matrix_UUt + self.damping[self.cov_step] * self.ops_eye(self.batch_size,self.batch_size,ms.float32)
        tmp0 = self.cholesky(damped_UUt)
        self.matrix_iUUt = self.matmul_at(tmp0, tmp0)

        return dout

    def construct(self, x):
        if self.seng:
            self.input_tensor = x

            output = self.conv2d(x, self.weight)
            output = self.getG(output)

            ### Code irrelevant to our algorithm. Just to generate the graph.
            output = ms.ops.depend(output,  ms.ops.assign(self.fake_current_damping, self.damping[self.cov_step]))
        else:
            output = self.conv2d(x, self.weight)

        return output

    def infer_shape(self):
        return self.conv2d_infer_shape(self.conv2d, self.batched_input_shape, self.weight_shape, self.bias_shape)

    # I wonder why this is removed in mindspore 1.2. polyfill.
    def conv2d_infer_shape(self, conv2d, x_shape, w_shape, b_shape=None):
        x_shape_norm = x_shape
        w_shape_norm = w_shape

        kernel_size_h = w_shape_norm[2]
        kernel_size_w = w_shape_norm[3]

        stride_h = conv2d.stride[2]
        stride_w = conv2d.stride[3]
        dilation_h = conv2d.dilation[2]
        dilation_w = conv2d.dilation[3]

        if conv2d.pad_mode == "valid":
            h_out = math.ceil((x_shape_norm[2] - dilation_h * (kernel_size_h - 1)) / stride_h)
            w_out = math.ceil((x_shape_norm[3] - dilation_w * (kernel_size_w - 1)) / stride_w)
            # pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
        elif conv2d.pad_mode == "same":
            h_out = math.ceil(x_shape_norm[2] / stride_h)
            w_out = math.ceil(x_shape_norm[3] / stride_w)

            # pad_needed_h = max(0, (h_out - 1) * stride_h + dilation_h * (kernel_size_h - 1) + 1 - x_shape_norm[2])
            # pad_top = math.floor(pad_needed_h / 2)
            # pad_bottom = pad_needed_h - pad_top

            # pad_needed_w = max(0, (w_out - 1) * stride_w + dilation_w * (kernel_size_w - 1) + 1 - x_shape_norm[3])
            # pad_left = math.floor(pad_needed_w / 2)
            # pad_right = pad_needed_w - pad_left
        elif conv2d.pad_mode == 'pad':
            # pad_top, pad_bottom, pad_left, pad_right = conv2d.padding

            h_out = 1 + (x_shape_norm[2] + pad_top + pad_bottom - kernel_size_h - (kernel_size_h - 1) \
                         * (dilation_h - 1)) / stride_h
            w_out = 1 + (x_shape_norm[3] + pad_left + pad_right - kernel_size_w - (kernel_size_w - 1) \
                         * (dilation_w - 1)) / stride_w
            h_out = math.floor(h_out)
            w_out = math.floor(w_out)

        out_channel = conv2d.out_channel
        out_shape = [x_shape_norm[0], out_channel, h_out, w_out]
        return out_shape

    def infer_out_hw(self):
        batched_out_shape = self.infer_shape()
        return batched_out_shape[2], batched_out_shape[3]

    def extra_repr(self):
        s = 'input_channels={}, output_channels={}, kernel_size={},' \
            'stride={},  pad_mode={}, padding={}, dilation={}, ' \
            'group={}, data_format={}, has_bias={}'.format(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.pad_mode,
                self.padding,
                self.dilation,
                self.group,
                self.data_format,
                self.has_bias)
        return s


class Dense_SENG_GPU(ms.nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init,
                 bias_init,
                 damping,
                 has_bias,
                 activation,
                 extra_args):
        super(Dense_SENG_GPU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.has_bias = has_bias
        self.seng = extra_args.is_train
        if isinstance(weight_init, ms.Tensor):
            if weight_init.dim() != 2 or weight_init.shape[0] != out_channels or \
                    weight_init.shape[1] != in_channels:
                raise ValueError("weight_init shape error")

        self.weight = ms.Parameter(ms.common.initializer.initializer(weight_init, [out_channels, in_channels]), name="weight")

        if self.has_bias:
            if isinstance(bias_init, ms.Tensor):
                if bias_init.dim() != 1 or bias_init.shape[0] != out_channels:
                    raise ValueError("bias_init shape error")

            self.bias = ms.Parameter(ms.common.initializer.initializer(bias_init, [out_channels]), name="bias")

        batch_size = extra_args.batch_size
        frequency = extra_args.frequency
        loss_scale = extra_args.loss_scale
        col_sample_size = extra_args.col_sample_size
        im_size_threshold = extra_args.im_size_threshold

        self.batch_size = batch_size
        self.batched_input_shape = (batch_size, in_channels)

        self.matmul_bt = ms.ops.MatMul(transpose_b=True)
        self.bias_add = ms.ops.BiasAdd()

        self.activation = ms.nn.layer.get_activation(activation)
        matrix_A_shape = (batch_size, in_channels, 1)
        matrix_G_shape = (batch_size, out_channels, 1)
        parameter_numel = out_channels * in_channels
        self.matrix_U_shape = (batch_size, parameter_numel)
        self.freq = ms.Tensor(frequency, ms.int32)

        # self.matrix_U = ms.Parameter(np.zeros(self.matrix_U_shape).astype(np.float32), name='matrix_U', requires_grad=False)
        self.input_tensor = ms.Parameter(np.zeros(self.batched_input_shape).astype(np.float32), name='input_tensor', requires_grad=False)
        self.matrix_iUUt = ms.Parameter(np.zeros((batch_size, batch_size)).astype(np.float32), name='matrix_iUUt', requires_grad=False)
        self.cov_step = ms.Parameter(ms.common.initializer.initializer(0, [1], ms.int32), name="cov_step", requires_grad=False)
        self.damping = ms.Parameter(ms.Tensor(damping, dtype=ms.float32), name="damping_value", requires_grad=False)
        self.fake_current_damping = ms.Parameter(np.zeros((1,)).astype(np.float32), name="fake_current_damping", requires_grad=False)

        is_implicit_representation = (im_size_threshold < parameter_numel)
        self.is_implicit_representation = is_implicit_representation
        if self.is_implicit_representation:
            self.is_sample_A = col_sample_size < matrix_A_shape[1]
            self.is_sample_G = col_sample_size < matrix_G_shape[1]
            self.is_col_sample = self.is_sample_A or self.is_sample_G

            self.matrix_A =  ms.Parameter(np.zeros(matrix_A_shape, dtype=np.float32), name='matrix_A', requires_grad=False)
            self.matrix_G = ms.Parameter(np.zeros(matrix_G_shape, dtype=np.float32), name='matrix_G', requires_grad=False)
            if self.is_col_sample:
                self.matrix_U_all_index = ms.Tensor(np.arange(parameter_numel, dtype=np.float16).reshape(matrix_G_shape[1], matrix_A_shape[1]))

                self.matrix_sub_G_shape = (matrix_G_shape[0], min(matrix_G_shape[1], col_sample_size), matrix_G_shape[2])
                self.matrix_sub_A_shape = (matrix_A_shape[0], min(matrix_A_shape[1], col_sample_size), matrix_A_shape[2])

                real_col_sample_size = self.matrix_sub_G_shape[1] * self.matrix_sub_A_shape[1]

                self.matrix_sub_A = ms.Parameter(np.zeros(self.matrix_sub_A_shape).astype(np.float32), name='matrix_sub_A', requires_grad=False)
                self.matrix_sub_G = ms.Parameter(np.zeros(self.matrix_sub_G_shape).astype(np.float32), name='matrix_sub_G', requires_grad=False)
                self.sample_index = ms.Parameter(np.zeros(real_col_sample_size).astype(np.int32), name='sample_index', requires_grad=False)

            else:
                # fake param, ensuring that every parameter is defined in every execution path
                self.matrix_sub_A = ms.Parameter(np.zeros((1, 1, 1)).astype(np.float32), name='matrix_sub_A', requires_grad=False)
                self.matrix_sub_G = ms.Parameter(np.zeros((1, 1, 1)).astype(np.float32), name='matrix_sub_G', requires_grad=False)
                self.sample_index = ms.Parameter(np.zeros((1,)).astype(np.int32), name='sample_index', requires_grad=False)

            if self.is_sample_A:
                self.rnd_choice_mask_A = ms.ops.RandomChoiceWithMask(count=col_sample_size)
                self.cols_as_mask_A = ms.Tensor(np.ones(shape=[self.matrix_sub_A_shape[1]]).astype(np.bool))
                self.mul_ratio_A = ms.Tensor(self.matrix_sub_A_shape[1] / col_sample_size, ms.float32)
            else:
                self.sample_index_A = ms.Tensor(np.arange(matrix_A_shape[1], dtype=np.int32))

            if self.is_sample_G:
                self.rnd_choice_mask_G = ms.ops.RandomChoiceWithMask(count=col_sample_size)
                self.cols_as_mask_G = ms.Tensor(np.ones(shape=[self.matrix_sub_G_shape[1]]).astype(np.bool))
                self.mul_ratio_G = ms.Tensor(self.matrix_sub_G_shape[1] / col_sample_size, ms.float32)
            else:
                self.sample_index_G = ms.Tensor(np.arange(matrix_G_shape[1], dtype=np.int32))
            # fake param
            self.matrix_sub_U = ms.Parameter(np.zeros((1,1), dtype=np.float32), name='matrix_sub_U', requires_grad=False)
            self.matrix_U = ms.Parameter(np.zeros((1,1), dtype=np.float32), name='matrix_U', requires_grad=False)
        else:
            # explicit U
            self.matrix_U = ms.Parameter(np.zeros(self.matrix_U_shape, dtype=np.float32), name='matrix_U', requires_grad=False)
            real_col_sample_size = col_sample_size * col_sample_size
            self.is_col_sample = real_col_sample_size < parameter_numel

            if self.is_col_sample:
                self.matrix_sub_U_shape = (batch_size, real_col_sample_size)
                self.matrix_sub_U = ms.Parameter(np.zeros(self.matrix_sub_U_shape, dtype=np.float32), name='matrix_sub_U', requires_grad=False)
                self.sample_index = ms.Parameter(np.zeros(real_col_sample_size, dtype=np.int32), name='sample_index', requires_grad=False)
                self.rnd_choice_mask = ms.ops.RandomChoiceWithMask(count=real_col_sample_size)
                self.cols_as_mask = ms.Tensor(np.ones(shape=[parameter_numel], dtype=np.bool))

                self.mul_ratio = ms.Tensor(parameter_numel / real_col_sample_size, ms.float32)
            else:
                # fake param
                self.matrix_sub_U = ms.Parameter(np.zeros((1,1), dtype=np.float32), name='matrix_sub_U', requires_grad=False)
                self.sample_index = ms.Parameter(np.zeros(1, dtype=np.int32), name='sample_index', requires_grad=False)
            # fake param
            self.matrix_A = ms.Parameter(np.zeros((1, 1, 1), dtype=np.float32), name='matrix_A', requires_grad=False)
            self.matrix_G = ms.Parameter(np.zeros((1, 1, 1), dtype=np.float32), name='matrix_G', requires_grad=False)
            self.matrix_sub_A = ms.Parameter(np.zeros((1,1,1), dtype=np.float32), name='matrix_sub_A', requires_grad=False)
            self.matrix_sub_G = ms.Parameter(np.zeros((1,1,1), dtype=np.float32), name='matrix_sub_G', requires_grad=False)

        # For train.py to test layer type
        layer_seng_type = 1
        if self.is_col_sample:
            layer_seng_type += 2
        if is_implicit_representation:
            layer_seng_type += 4
        self.layer_seng_type = ms.Parameter(np.zeros(layer_seng_type).astype(np.bool), name='layer_seng_type', requires_grad=False)

        self.matrix_G_normalizer = ms.Tensor(batch_size**0.5 / loss_scale, ms.float32)

        self.gather = ms.ops.Gather()
        self.matmul_at = ms.ops.MatMul(transpose_a=True)
        self.bmm_bt = ms.ops.BatchMatMul(transpose_b=True)
        self.getG = ms.ops.InsertGradientOf(self.save_gradient)
        self.ops_eye = ms.ops.Eye()
        self.cholesky = ms.ops.operations.CholeskyTrsm()


    def save_gradient(self, dout):
        matrix_A = ms.ops.expand_dims(ms.ops.cast(self.input_tensor, ms.float32), 2)
        matrix_G = ms.ops.expand_dims(self.matrix_G_normalizer * ms.ops.cast(dout, ms.float32), 2)

        if self.is_implicit_representation:
            self.matrix_A = matrix_A
            self.matrix_G = matrix_G
            if self.is_col_sample:
                if self.is_sample_A:
                    sample_index_A = self.rnd_choice_mask_A(self.cols_as_mask_A)[0][:,0]
                    matrix_sub_A = self.mul_ratio_A * self.gather(matrix_A, sample_index_A, 1)
                    sample_index = self.gather(self.matrix_U_all_index, sample_index_A, 1)
                else:
                    sample_index_A = self.sample_index_A
                    matrix_sub_A = matrix_A
                    sample_index = self.matrix_U_all_index

                if self.is_sample_G:
                    sample_index_G = self.rnd_choice_mask_G(self.cols_as_mask_G)[0][:,0]
                    matrix_sub_G = self.mul_ratio_G * self.gather(matrix_G, sample_index_G, 1)
                    sample_index = self.gather(sample_index, sample_index_G, 0)
                else:
                    sample_index_G = self.sample_index_G
                    matrix_sub_G = matrix_G

                sample_index = ms.ops.reshape(sample_index, (-1,))
                sample_index = ms.ops.cast(sample_index, ms.int32)
                self.sample_index = sample_index
                self.matrix_sub_A = matrix_sub_A
                self.matrix_sub_G = matrix_sub_G
            else:
                matrix_sub_A = matrix_A
                matrix_sub_G = matrix_G

            matrix_UUt = self.matmul_bt(matrix_sub_A[:,:,0], matrix_sub_A[:,:,0]) *  self.matmul_bt(matrix_sub_G[:,:,0], matrix_sub_G[:,:,0])
        else:
            matrix_U = ms.ops.reshape(self.bmm_bt(matrix_G, matrix_A), (self.batch_size, -1))
            self.matrix_U = matrix_U
            if self.is_col_sample:
                sample_index = self.rnd_choice_mask(self.cols_as_mask)[0][:,0]
                matrix_sub_U = self.mul_ratio * self.gather(matrix_U, sample_index, 1)
                self.sample_index = sample_index
                self.matrix_sub_U = matrix_sub_U
            else:
                matrix_sub_U = matrix_U
            matrix_UUt = self.matmul_bt(matrix_sub_U, matrix_sub_U)

        self.cov_step = self.cov_step + self.freq
        damped_UUt = matrix_UUt + self.damping[self.cov_step] * self.ops_eye(self.batch_size,self.batch_size,ms.float32)
        tmp0 = self.cholesky(damped_UUt)
        self.matrix_iUUt = self.matmul_at(tmp0, tmp0)
        return dout

    def construct(self, x):
        if self.seng:
            self.input_tensor = x
            output = self.matmul_bt(x, self.weight)
            output = self.getG(output)

            ### Code irrelevant to our algorithm. Just to generate the graph
            output = ms.ops.depend(output, ms.ops.assign(self.fake_current_damping, self.damping[self.cov_step]))
        else:
            output = self.matmul_bt(x, self.weight)

        if self.has_bias:
            output = self.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def infer_shape(self):
        return (self.batch_size, self.out_channels)

    def extend_repr(self):
        """extend_repr"""
        str_info = 'in_channels={}, out_channels={}, weight={}, has_bias={}' \
            .format(self.in_channels, self.out_channels, self.weight, self.has_bias)
        if self.has_bias:
            str_info = str_info + ', bias={}'.format(self.bias)

        if self.activation is not None:
            str_info = str_info + ', activation={}'.format(self.activation)

        return str_info
