import numpy as np
import mindspore as ms
import mindspore.profiler

# from resnet import resnet50
from resnet_seng import resnet50
from model_seng import Model_SENG
from lr_generator import get_lr
from seng import SENG_GPU
from checkpoint_seng import CheckPointSENG
from utils import create_cifar10_dataset, create_ilsvrc2012_dataset, get_seng_parser, AccMonitor, next_profiler_dir, preprocess_args, SmoothCrossEntropy
from step_time_monitor import StepTimeMonitor

def get_model_damping(args,total_epochs,steps_per_epoch):
    damping_each_step = []
    total_epochs = int(total_epochs)
    total_steps = steps_per_epoch * total_epochs
    for i in range(total_steps):
        tmp0 = (i+1) / steps_per_epoch
        damping_local = args.damping_init*(args.damping_decay ** (tmp0/10))
        damping_each_step.append(damping_local)
    damping_each_step = np.array(damping_each_step).astype(np.float32)
    return damping_each_step

ms.common.set_seed(1)

if __name__ == '__main__':
    tmp0 = ('--data_path /mnt/cifar10 --debug 2 --epoch_size 1 --lr_init 0.01 --lr_max 0.1 --lr_end 0.0001 '
        '--batch_size 128 --damping_init 0.5 --damping_decay 0.5 --momentum 0.9 --warmup_epoch 0 --loss_scale 1024 '
        '--weight_decay 5e-4 --frequency 200 --im_size_threshold 1000000 --save_ckpt 0 --verbose 1 --image_size 224 '
        '--ckpt_save_path /userhome/project/ms-seng/ckpt_debug')
    # args = get_seng_parser().parse_args(tmp0.split()) #so we can run it in ipython
    args = get_seng_parser().parse_args()
    preprocess_args(args)

    save_graphs = (args.debug & 1 != 0)
    do_profile = (args.debug & 2 != 0)

    ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target=args.device_target,
        save_graphs=save_graphs, max_call_depth=10000, enable_graph_kernel=True)

    if do_profile:
        profiler = ms.profiler.Profiler(output_path=next_profiler_dir('seng'))

    if args.device_num > 1:
        ms.communication.management.init()
        ms.context.set_auto_parallel_context(device_num=args.device_num,
            parallel_mode=ms.context.ParallelMode.DATA_PARALLEL, gradients_mean=True)
        rank = ms.communication.management.get_rank()
        # prevent different process from writing on the same ckpt
        args.ckpt_save_path = args.ckpt_save_path + "_" + str(rank) + "/"
    else:
        rank = 0

    image_size = args.image_size
    if args.dataset == "cifar10":
        ds_train = create_cifar10_dataset(args.data_path, 'train', batch_size=args.batch_size, image_size=image_size, repeat_size=1)
    else:
        ds_train = create_ilsvrc2012_dataset(args.data_path, 'train', batch_size=args.batch_size, image_size=image_size, repeat_size=1, world_size=args.device_num, rank=rank)
    # ds_val = create_cifar10_dataset(args.data_path, 'val', batch_size=args.batch_size)
    len_ds_train = ds_train.get_dataset_size()
    lr = get_lr(lr_init=args.lr_init, lr_end=args.lr_end, lr_max=args.lr_max, warmup_epochs=args.warmup_epoch,
                total_epochs=args.epoch_size, steps_per_epoch=len_ds_train, lr_decay_mode=args.lr_decay_mode, decay_epochs=args.decay_epochs)
    damping = get_model_damping(args, total_epochs=args.epoch_size, steps_per_epoch=len_ds_train)
    # damping = get_model_damping(0, args.damping_init, args.damping_decay, args.epoch_size, len_ds_train)

    net = resnet50(args.class_num, damping=damping, input_hw=(image_size,image_size), extra_args=args)

    all_parameters = tuple(net.get_parameters())
    opt = SENG_GPU(net.trainable_params(), ms.Tensor(lr), args.momentum, all_parameters, damping, args.loss_scale, args)

    # loss = ms.nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    loss = SmoothCrossEntropy(args.label_smoothing, args.class_num)
    loss_scale = ms.train.loss_scale_manager.FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)
    model_args = {"loss_fn": loss, "optimizer": opt, "loss_scale_manager": loss_scale,
      "metrics": {'acc1':ms.nn.metrics.Accuracy()}, "amp_level": "O2", "keep_batchnorm_fp32": False}
    if args.frequency == 1:
        # This part enables debugging with single graph.
        model = ms.train.Model(net, **model_args)
    else:
        model = Model_SENG(net, frequency=args.frequency, **model_args)

    if args.verbose >= 1:
        callbacks = [ms.train.callback.LossMonitor(), StepTimeMonitor()]
    else:
        callbacks = [ms.train.callback.TimeMonitor()]

    # , AccMonitor(model, ds_val)
    if args.save_ckpt:
        config_ck = ms.train.callback.CheckpointConfig(save_checkpoint_steps=len_ds_train, keep_checkpoint_max=args.epoch_size)
        ckpoint_cb = CheckPointSENG(prefix="checkpoint_seng", config=config_ck, directory=args.ckpt_save_path)
        callbacks.append(ckpoint_cb)

    # dataset_sink_mode must be True here, otherwise ./dataset_helper.py will raise Error
    model.train(args.epoch_size, ds_train, callbacks=callbacks, dataset_sink_mode=True)
    if do_profile:
        profiler.analyse()


# zc0 = np.logical_not(np.array(opt.layer_is_implicit))
# zc1 = np.logical_not(np.array(opt.layer_is_sample))
# zc2 = np.where(np.logical_and(zc0, zc1))[0].tolist()
# zc3 = dict(list(net.cells_and_names()))
# zc4 = [zc3[opt.parameters[3*x].name.rsplit('.',1)[0]] for x in zc2]

# from itertools import groupby
# from utils import calculate_model_size, hf_ms_size
# z0 = calculate_model_size(net, train=False)
# z1 = sorted([(x.name.rsplit('.',1)[1], hf_ms_size(x)) for x in z0], key=lambda x:x[0])
# z2 = [(x,sum(z for _,z in y)) for x,y in groupby(z1, key=lambda x:x[0])]
# sorted([(x,sum(z for _,z in y)) for x,y in groupby(z1, key=lambda x:x[0])], key=lambda x:x[1])[::-1]

# z0 = dict(list(net.cells_and_names()))
# z1 = [z0[x.name.rsplit('.',1)[0]] for x in opt.matrix_A]
# for ind0,x in enumerate(z1):
#     print(ind0, x.weight.name)
#     _ = x.save_gradient(ms.Tensor(np.random.randn(*x.infer_shape()),dtype=ms.float32))
# for x in z1:
#     print(x.weight.name, x.infer_shape(), sep=' $ ')

# from utils import compare_two_profile
# _ = compare_two_profile('ms_profiler/20210113-1716_3209_sgd', 'ms_profiler/20210113-1712_235_seng')
