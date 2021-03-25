import argparse
import os
import re

import mindspore.nn as nn
from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore import context
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from utils import create_cifar10_dataset, create_ilsvrc2012_dataset, preprocess_args, SmoothCrossEntropy
from resnet_sgd import resnet50

def get_epoch_from_filename(filename):
    match = re.search(r'(\d+)_\d+.ckpt', filename)
    return int(match.group(1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ResNet with KFAC/SENG/SGD on Mindspore')
    parser.add_argument('--device_target', type=str, default="GPU", choices=['Ascend', 'GPU', 'CPU'])
    parser.add_argument('--dataset', type=str, default="cifar10", choices=["cifar10", "imagenet2012"])
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--class_num', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_path', type=str, default='/userhome/datasets/cifar-10/cifar-10-verify-bin')
    parser.add_argument('--is_train', type=int, default=0)
    parser.add_argument('--label_smoothing', type=float, default=0.0)

    # fake args supplied for seng, seng_layer and utils
    parser.add_argument('--image_size', type=int, default=0)
    parser.add_argument('--frequency', type=int, default=1)
    parser.add_argument('--use_nesterov', type=int, default=0)
    parser.add_argument('--loss_scale', type=float, default=1.0)
    parser.add_argument('--col_sample_size', type=int, default=128)
    parser.add_argument('--im_size_threshold', type=int, default=1000000)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--device_num', type=int, default=1)
    args = parser.parse_args()
    preprocess_args(args)

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=False)
    dataset_sink_mode = False

    image_size = args.image_size
    tag_cifar = (image_size == 32)
    if args.is_train:
        phase = 'train'
    else:
        phase = 'val'
    net = resnet50(class_num=args.class_num, tag_cifar=tag_cifar)
    if args.dataset == "cifar10":
        ds_eval = create_cifar10_dataset(args.data_path, phase, batch_size=args.batch_size, image_size=image_size, repeat_size=1)
    else:
        ds_eval = create_ilsvrc2012_dataset(args.data_path, phase, batch_size=args.batch_size, image_size=image_size, repeat_size=1)
    ckpts = [x for x in os.listdir(args.ckpt_path) if x.endswith('.ckpt')]
    ckpts.sort(key=get_epoch_from_filename)
    for ckpt in ckpts:
        param_dict = load_checkpoint(os.path.join(args.ckpt_path, ckpt))
        load_param_into_net(net, param_dict)
        net.set_train(False)
        # loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        loss = SmoothCrossEntropy(args.label_smoothing, args.class_num)
        model = Model(net, loss, metrics={'top_1_accuracy', 'loss'})
        eval_res = model.eval(ds_eval, dataset_sink_mode=dataset_sink_mode)
        print("eval: {:d}  {:6.3f}  {:6.4f}".format(
            get_epoch_from_filename(ckpt), eval_res['top_1_accuracy'] * 100, eval_res['loss']), flush=True)
