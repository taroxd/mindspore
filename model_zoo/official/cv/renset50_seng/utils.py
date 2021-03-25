import os
import argparse
import datetime
import numpy as np
import pandas as pd
from itertools import groupby
import mindspore as ms
import mindspore.dataset.transforms

class SmoothCrossEntropy(ms.nn.loss.loss._Loss):
    def __init__(self, smooth_factor=0, num_classes=1000):
        super().__init__()
        self.onehot = ms.ops.OneHot()
        self.on_value = ms.Tensor(1 - smooth_factor, ms.float32)
        self.off_value = ms.Tensor(smooth_factor / (num_classes - 1), ms.float32)
        self.ce = ms.nn.SoftmaxCrossEntropyWithLogits(reduction='mean')
        # self.mean = ms.ops.ReduceMean(False)

    def construct(self, logit, label):
        one_hot_label = self.onehot(label, ms.ops.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, one_hot_label)
        # loss = self.mean(loss, 0)
        return loss


def create_cifar10_dataset(data_path, phase, image_size=32, batch_size=32, repeat_size=1):
    assert phase in {'train','val'}
    is_training = phase=='train'
    op_image = [ms.dataset.transforms.c_transforms.TypeCast(ms.float32)]
    if is_training:
        op_image += [
            ms.dataset.vision.c_transforms.RandomCrop((32, 32), (4, 4, 4, 4)),
            ms.dataset.vision.c_transforms.RandomHorizontalFlip(prob=0.5)
        ]
    if image_size!=32:
        op_image.append(ms.dataset.vision.c_transforms.Resize(image_size))
    op_image += [
        ms.dataset.vision.c_transforms.Rescale(1/255, 0),
        ms.dataset.vision.c_transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ms.dataset.vision.c_transforms.HWC2CHW()
    ]
    op_label = [ms.dataset.transforms.c_transforms.TypeCast(ms.int32)]

    if is_training:
        ds = ms.dataset.Cifar10Dataset(data_path, 'train', shuffle=True)
    else:
        ds = ms.dataset.Cifar10Dataset(data_path, 'test', shuffle=False)
    ds = ds.map(operations=op_label, input_columns="label")
    ds = ds.map(operations=op_image, input_columns="image")
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat(repeat_size)
    return ds


def create_ilsvrc2012_dataset(dataset_path, phase, batch_size=32, image_size=224, min_scale=0.08, repeat_size=1, world_size=1, rank=0):
    assert phase in {'train','val'}
    is_training = phase=='train'

    ds = ms.dataset.ImageFolderDataset(dataset_path, num_parallel_workers=8,
                shuffle=is_training, num_shards=world_size, shard_id=rank)

    mean = [0.485*255, 0.456*255, 0.406*255]
    std = [0.229*255, 0.224*255, 0.225*255]

    if is_training:
        op_image = [
            ms.dataset.vision.c_transforms.RandomCropDecodeResize(image_size, scale=(min_scale,1), ratio=(0.75,1.333)),
            ms.dataset.vision.c_transforms.RandomHorizontalFlip(prob=0.5),
            ms.dataset.vision.c_transforms.Normalize(mean=mean, std=std),
            ms.dataset.vision.c_transforms.HWC2CHW()
        ]
    else:
        op_image = [
            ms.dataset.vision.c_transforms.Decode(),
            ms.dataset.vision.c_transforms.Resize(256),
            ms.dataset.vision.c_transforms.CenterCrop(image_size),
            ms.dataset.vision.c_transforms.Normalize(mean=mean, std=std),
            ms.dataset.vision.c_transforms.HWC2CHW()
        ]
    op_label = [ms.dataset.transforms.c_transforms.TypeCast(ms.int32)]
    ds = ds.map(operations=op_image, input_columns="image", num_parallel_workers=8)
    ds = ds.map(operations=op_label, input_columns="label", num_parallel_workers=8)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat(repeat_size)
    return ds


class AccMonitor(ms.train.callback.Callback):
    def __init__(self, model, ds_val, dataset_sink_mode=True):
        super(AccMonitor, self).__init__()
        self.model = model
        self.ds_val = ds_val
        self.dataset_sink_mode = dataset_sink_mode
        self.len_dataset = ds_val.get_dataset_size() * ds_val.get_batch_size() #TODO bad

    def epoch_end(self, run_context):
        tmp0 = self.model.eval(self.ds_val, dataset_sink_mode=self.dataset_sink_mode)['acc1']
        print(f'[validation] acc1={tmp0}')


def get_sgd_parser():
    parser = argparse.ArgumentParser(description='ResNet with KFAC/SENG/SGD on Mindspore')
    parser.add_argument('--dataset', type=str, default="cifar10", choices=["cifar10", "imagenet2012"])
    parser.add_argument('--class_num', type=int, default=10)
    parser.add_argument('--device_target', type=str, default="GPU", choices=['Ascend', 'GPU'])
    parser.add_argument('--epoch_size', type=int, default=90)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_init', type=float, default=0.01)
    parser.add_argument('--lr_max', type=float, default=0.1)
    parser.add_argument('--lr_end', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--loss_scale', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--data_path', type=str, default='/userhome/datasets/cifar-10/cifar-10-batches-bin')
    parser.add_argument('--save_ckpt', type=int, default=1)
    parser.add_argument('--ckpt_save_path', type=str, default="./ckpt")
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--debug', type=int, default=0)
    return parser


def get_seng_parser():
    parser = argparse.ArgumentParser(description='ResNet with KFAC/SENG/SGD on Mindspore')
    parser.add_argument('--dataset', type=str, default="cifar10", choices=["cifar10", "imagenet2012"])
    parser.add_argument('--class_num', type=int, default=0, help='Legacy argument. Set in preprocess code.')
    parser.add_argument('--device_target', type=str, default="GPU", choices=['Ascend', 'GPU'])
    parser.add_argument('--epoch_size', type=int, default=90)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_init', type=float, default=0.01)
    parser.add_argument('--lr_max', type=float, default=0.1)
    parser.add_argument('--lr_end', type=float, default=0.0001)
    parser.add_argument('--lr_decay_mode', type=str, default="steps", choices=['steps', 'poly', 'cosine', 'linear', 'exp'])
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--damping_init', type=float, default=0.1)
    parser.add_argument('--damping_decay', type=float, default=0.2)
    parser.add_argument('--decay_epochs', type=int, default=40)
    parser.add_argument('--damping_keep_const_in_optimizer', type=int, default=0,
        help='Keep the damping in the same step when scaling the gradients. May yield a better result.')
    parser.add_argument('--frequency', type=int, default=1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--use_nesterov', type=int, default=0)
    parser.add_argument('--warmup_epoch', type=int, default=0)
    parser.add_argument('--loss_scale', type=float, default=128.0)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--data_path', type=str, default='/userhome/datasets/cifar-10/cifar-10-batches-bin')
    parser.add_argument('--save_ckpt', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=0, help='Input image size (only effective for cifar10)')
    parser.add_argument('--col_sample_size', type=int, default=128)
    parser.add_argument('--im_size_threshold', type=int, default=1000000)
    parser.add_argument('--ckpt_save_path', type=str, default="./ckpt")
    parser.add_argument('--device_num', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--debug', type=int, default=0, help="bitflag, 1: save_graphs  2: profile")
    parser.add_argument('--is_train', type=int, default=1, help="Fake args. Do not change its value")
    return parser

def preprocess_args(args):
    if args.dataset == 'imagenet2012':
        args.class_num = 1000
        args.image_size = 224
    else:
        args.class_num = 10
        if args.image_size == 0:
            args.image_size = 32

def next_profiler_dir(key=None, ROOT_DIR='ms_profiler'):
    if not os.path.exists(ROOT_DIR):
        os.makedirs(ROOT_DIR)
    tmp0 = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    tmp1 = str(np.random.randint(10000))
    if key is None:
        name = f'{tmp0}_{tmp1}'
    else:
        name = f'{tmp0}_{tmp1}_{key}'
    ret = os.path.join(ROOT_DIR, name)
    assert not os.path.exists(ret)
    os.makedirs(ret)
    return ret


def hf_ms_size(x):
    dtype_to_byte = {'Float32':4, 'Int32':4, 'Bool':1}
    ret = np.prod(x.shape)*dtype_to_byte[str(x.dtype)]/2**30
    return ret

def calculate_model_size(net, train=True):
    # dtype_to_byte = {'Float32':4, 'Int32':4, 'Bool':1}
    # hf_ms_size = lambda x: np.prod(x.shape)*dtype_to_byte[str(x.dtype)]/2**30
    if train:
        z0 = net.trainable_params()
    else:
        z0 = list(net.get_parameters())
    total_size = sum([hf_ms_size(x) for x in z0])
    print(f'size(GB): {total_size}')
    return z0


def parse_profile(filename):
    # filename = 'ms_profiler/20210113-1716_3209_sgd/profiler/gpu_op_detail_info_0.csv'
    if not filename.endswith('.csv'):
        filename = os.path.join(filename, 'profiler', 'gpu_op_detail_info_0.csv')
    assert os.path.exists(filename)
    pd0 = pd.read_csv(filename)
    tmp0 = list(zip(pd0['op_type'].values.tolist(), pd0['op_total_time(us)'].values.tolist()))
    tmp1 = groupby(sorted(tmp0, key=lambda x:x[0]), key=lambda x:x[0])
    ret0 = {x:sum(z for _,z in y)/10**6 for x,y in tmp1}
    return ret0, pd0


def compare_two_profile(filepath0, filepath1):
    z0,pd0 = parse_profile(filepath0)
    z1,pd1 = parse_profile(filepath1)

    z2 = dict()
    for k in set(z0.keys()) | set(z1.keys()):
        tmp0 = float(z0.get(k,0))
        tmp1 = float(z1.get(k,0))
        tmp2 = tmp1 - tmp0
        z2[k] = (tmp0,tmp1,tmp2)

    tmp0 = sum(x[0] for x in z2.values())
    tmp1 = sum(x[1] for x in z2.values())
    tmp2 = tmp1 - tmp0
    z2['TOTAL'] = (tmp0,tmp1,tmp2)

    z2 = sorted([(k,x,y,z) for k,(x,y,z) in z2.items()], key=lambda x:x[3])[::-1]
    for (k,x,y,z) in z2:
        print(f'| {k:37} | {x:7.3} | {y:7.3} | {z:7.3} |')
    return z0,z1,pd0,pd1
