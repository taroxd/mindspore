import numpy as np
import mindspore as ms

_momentum_opt = ms.ops.MultitypeFuncGraph("momentum_opt")

op_add = ms.ops.operations.AddN()
apply_decay = ms.ops.MultitypeFuncGraph("apply_decay")

@apply_decay.register("Number", "Bool", "Tensor", "Tensor")
def _tensor_apply_decay(weight_decay, if_apply, weight, gradient):
    if if_apply:
        # use op_add for fusion of weight decay and momentum
        # weight needs to be placed before weight_decay
        return op_add((weight*weight_decay, gradient))
    return gradient


@_momentum_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt_ext(opt, momentum, learning_rate, gradient, weight, moment):
    """Apply momentum optimizer to the weight parameter using Tensor."""
    success = ms.ops.depend(True, opt(weight, moment, learning_rate, gradient, momentum))
    # opt(weight, moment, learning_rate, gradient, momentum) # seems depend is useless
    return success

class SENG_GPU(ms.nn.optim.Optimizer):
    def __init__(self, params, learning_rate, momentum, all_parameters, damping, loss_scale, extra_args, decay_exclude=None):
        weight_decay = extra_args.weight_decay
        col_sample_size = extra_args.col_sample_size
        batch_size = extra_args.batch_size
        use_nesterov = (extra_args.use_nesterov != 0)
        self.damping_keep_const = False
        if hasattr(extra_args, 'damping_keep_const_in_optimizer') and extra_args.damping_keep_const_in_optimizer:
            self.damping_keep_const = True

        self.seng = extra_args.is_train

        super(SENG_GPU, self).__init__(learning_rate, params, weight_decay, loss_scale)
        if momentum < 0:
            raise ValueError("momentum should be at least 0.0, but got momentum {}".format(momentum))

        self.momentum = ms.Parameter(ms.Tensor(momentum, ms.float32), name="momentum")
        self.moments = self.parameters.clone(prefix="moments", init='zeros')
        self.hyper_map = ms.ops.HyperMap()
        self.opt = ms.ops.ApplyMomentum(use_nesterov=use_nesterov)
        self.col_sample_size = col_sample_size
        self.batch_size = batch_size
        self.transpose = ms.ops.Transpose()
        self.matmul = ms.ops.MatMul()
        self.matmul_at = ms.ops.MatMul(transpose_a=True)
        self.matmul_bt = ms.ops.MatMul(transpose_b=True)
        self.bmm = ms.ops.BatchMatMul()
        self.bmm_at = ms.ops.BatchMatMul(transpose_a=True)
        self.bmm_bt = ms.ops.BatchMatMul(transpose_b=True)
        # self.cholesky = ms.ops.operations.Cholesky(split_dim=0)
        self.cholesky = ms.ops.operations.Cholesky()
        self.abs = ms.ops.Abs()
        self.reduce_sum = ms.ops.ReduceSum(keep_dims=False)
        self.concat = ms.ops.Concat()
        self.all_reduce = ms.ops.AllReduce(ms.ops.operations.comm_ops.ReduceOp.SUM)

        hf0 = lambda name: [x for x in all_parameters if name in x.name]
        self.matrix_U = ms.ParameterTuple(hf0('matrix_U'))
        self.matrix_A = ms.ParameterTuple(hf0('matrix_A'))
        self.matrix_G = ms.ParameterTuple(hf0('matrix_G'))
        self.matrix_sub_U = ms.ParameterTuple(hf0('matrix_sub_U'))
        self.matrix_sub_A = ms.ParameterTuple(hf0('matrix_sub_A'))
        self.matrix_sub_G = ms.ParameterTuple(hf0('matrix_sub_G'))
        sample_index = hf0('sample_index')
        self.sample_index = ms.ParameterTuple(sample_index)
        self.matrix_iUUt = ms.ParameterTuple(hf0('matrix_iUUt'))
        layer_seng_type = [x.shape[0] for x in all_parameters if 'layer_seng_type' in x.name]

        self.layer_is_sample = [(x & 2 != 0) for x in layer_seng_type]
        self.layer_is_implicit = [(x & 4 != 0) for x in layer_seng_type]

        self.world_size = extra_args.device_num

        self.inv_damping = ms.Tensor(1/damping, dtype=ms.float32)

        self.bcast_gsub = [ms.ops.BroadcastTo((batch_size, x.shape[0])) for x in self.sample_index]

        self.mul_ratio = []
        for i in range(54):
            if self.layer_is_sample[i]:
                sample_size = self.sample_index[i].shape[0]
                g_numel = np.product(self.parameters[i * 3].shape)
                self.mul_ratio.append(ms.Tensor(g_numel / sample_size, ms.float32))
            else:
                self.mul_ratio.append(None)

        self.weight_decay = weight_decay
        if decay_exclude is None:
            decay_exclude = set()
        else:
            decay_exclude = set(decay_exclude)
        self.decay_flags = tuple((x.name not in decay_exclude) for x in self.parameters)

    def construct(self, gradients):
        params = self.parameters
        moments = self.moments
        lr = self.get_lr()

        gradients = self.scale_grad(gradients)

        if self.damping_keep_const:
            inv_damping = self.inv_damping[0]
        else:
            inv_damping = self.inv_damping[self.global_step]

        new_grads = ()
        for i in range(54):
            g = gradients[i * 3]
            g_orig_shape = g.shape
            g = inv_damping * ms.ops.reshape(g, (-1,1))

            if self.layer_is_sample[i]:
                g_sub = self.mul_ratio[i] * g[self.sample_index[i]]
            else:
                g_sub = g

            matrix_iUUt = ms.ops.depend(self.matrix_iUUt[i], g)
            if self.layer_is_implicit[i]:
                matrix_A = ms.ops.depend(self.matrix_A[i], g)
                matrix_G = ms.ops.depend(self.matrix_G[i], g)
                if self.layer_is_sample[i]:
                    matrix_sub_G = self.matrix_sub_G[i]
                    matrix_sub_A = self.matrix_sub_A[i]
                else:
                    matrix_sub_G = matrix_G
                    matrix_sub_A = matrix_A
                shape_G = matrix_G.shape
                shape_A = matrix_A.shape
                shape_sub_G = matrix_sub_G.shape
                shape_sub_A = matrix_sub_A.shape
                # U @ g
                # tmp = ms.ops.reshape(self.bmm_bt(matrix_sub_G, matrix_sub_A), (self.batch_size,-1))
                # tmp = self.matmul(tmp, ms.ops.reshape(g_sub, (-1,1)))
                # --- the code above is equivalent for U@g but seems to be a little slower ---
                tmp = ms.ops.reshape(g_sub, (-1,))
                tmp = self.bcast_gsub[i](tmp)
                tmp = ms.ops.reshape(tmp, (self.batch_size, shape_sub_G[1], shape_sub_A[1]))
                tmp = self.bmm(tmp, matrix_sub_A)
                tmp = tmp * matrix_sub_G
                tmp = self.reduce_sum(tmp, (1, 2))
                # inv(UUt) @ tmp
                tmp = ms.ops.reshape(tmp, (-1, 1))
                tmp = self.matmul(matrix_iUUt, tmp)
                # Ut @ tmp   ref: pytorch implementation (function _dwt_vector_product)
                coeff_g = ms.ops.sqrt(self.abs(tmp))
                coeff_a = tmp / (self.reduce_sum(coeff_g) + 1e-5)
                g_tmp = ms.ops.reshape(self.matmul_at(ms.ops.reshape(matrix_G, (self.batch_size,-1)), coeff_g), shape_G[1:])
                a_tmp = ms.ops.reshape(self.matmul_at(ms.ops.reshape(matrix_A, (self.batch_size,-1)), coeff_a), shape_A[1:])
                if self.world_size > 1:
                    # do the averaging only on a_tmp. Maybe helpful for reducing time of ops.Mul ?
                    g_tmp = self.all_reduce(g_tmp) # / self.world_size
                    a_tmp = self.all_reduce(a_tmp) / (self.world_size * self.world_size)
                g_second_term = ms.ops.reshape(self.matmul_bt(g_tmp,a_tmp), (-1, 1))
            else:
                matrix_U = ms.ops.depend(self.matrix_U[i], g)
                if self.layer_is_sample[i]:
                    matrix_sub_U = self.matrix_sub_U[i]
                else:
                    matrix_sub_U = matrix_U
                g_second_term = self.matmul_at(matrix_U, self.matmul(matrix_iUUt, self.matmul(matrix_sub_U, g_sub)))
                if self.world_size > 1:
                    g_second_term = self.all_reduce(g_second_term) / self.world_size
                # if self.layer_is_sample[i]:
                #     matrix_sub_U = ms.ops.depend(self.matrix_sub_U[i], g)
                #     tmp0 = self.matmul(self.matrix_iUUt[i], self.matmul(matrix_sub_U, g_sub))
                #     tmp1 = ms.ops.reshape(self.transpose(matrix_A, (1,2,0)), (matrix_A.shape[1],-1))
                #     tmp2 = ms.ops.reshape(self.transpose(matrix_G, (1,2,0))*tmp0[:,0], (matrix_G.shape[1],-1))
                #     g_second_term = ms.ops.reshape(self.matmul_bt(tmp2, tmp1), (-1,1))
                # else:
                #     tmp0 = ms.ops.reshape(g_sub,(matrix_G.shape[1],matrix_A.shape[1]))
                #     tmp1 = ms.ops.reshape(self.transpose(matrix_G,(0,2,1)), (-1,matrix_G.shape[1]))
                #     tmp2 = self.reduce_sum(ms.ops.reshape(self.matmul(tmp1,tmp0), matrix_A.shape)*matrix_A, (1,2))
                #     tmp3 = self.matmul(self.matrix_iUUt[i], ms.ops.reshape(tmp2,(-1,1)))
                #     tmp4 = ms.ops.reshape(self.transpose(matrix_G, (1,0,2))*tmp3, (matrix_G.shape[1],-1))
                #     tmp5 = ms.ops.reshape( self.transpose(matrix_A, (1,0,2)), (matrix_A.shape[1],-1))
                #     g_second_term = ms.ops.reshape(self.matmul_bt(tmp4, tmp5), (-1,1))
            g = g - g_second_term
            g = ms.ops.reshape(g, g_orig_shape)

            if i == 53:
                new_grads = new_grads + (g,)
            else:
                new_grads = new_grads + (g, gradients[i * 3 + 1], gradients[i * 3 + 2])
        gradients = new_grads

        if self.weight_decay > 0:
            gradients = self.hyper_map(ms.ops.partial(apply_decay, self.weight_decay), self.decay_flags, params, gradients)

        success = self.hyper_map(ms.ops.partial(_momentum_opt, self.opt, self.momentum, lr), gradients, params, moments)

        return success
