#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import copy
import os.path
import random
import socket
from functools import partial

import torch.distributed
import torch.utils.data
from timm.data.dataset import ImageDataset
from timm.utils import accuracy
from torch.utils.data import Dataset
from tqdm import tqdm

from quant import *
from utils import *
from utils.resnet import resnet101, resnet50, resnet18
from utils.utils import write, create_transform, create_loader, AverageMeter, broadcast_tensor_from_main_process, \
    gather_tensor_from_multi_processes, compute_quantized_params

HOST_NAME = socket.getfqdn(socket.gethostname())

torch.backends.cudnn.benchmark = True
LINEAR_COMPENSATION_SAMPLES = 512

model_path = {
    'resnet18': 'pretrained_weights/resnet18_imagenet.pth.tar',
    'resnet50': 'pretrained_weights/resnet50_imagenet.pth.tar',
    'resnet101': 'pretrained_weights/resnet101-63fe2227.pth'
}

def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

class CompensationBlock(nn.Module):
    def __init__(self, W, b, r2_score, block, groups, linear_init=True, local_rank=0, block_id=None):
        super(CompensationBlock, self).__init__()
        self.block = block
        self.groups = groups

        self.lora_weight = nn.Parameter(torch.zeros((W.size(0), W.size(1), W.size(2), W.size(3))))
        self.lora_bias = nn.Parameter(torch.zeros(b.size(0)))

        if linear_init and (r2_score > 0):
            self.lora_weight.data.copy_(W)
            self.lora_bias.data.copy_(b)
            if local_rank == 0:
                _write('block {} using linear init'.format(block_id))
        else:
            nn.init.zeros_(self.lora_weight)
            nn.init.zeros_(self.lora_bias)
            if local_rank == 0:
                _write('block {} using lora init'.format(block_id))

    def forward(self, x):
        out = self.block(x)

        B, C_X, H_X, W_X = x.size()
        _, C_Y, H_Y, W_Y = out.size()

        if (H_X == H_Y) and (W_X == W_Y):
            stride = 1
        elif (H_X // 2 == H_Y) and (W_X // 2 == W_Y):
            stride = 2
        else:
            raise NotImplementedError

        if self.training:
            qwt_out = F.conv2d(x, self.lora_weight, self.lora_bias, stride=stride, padding=int(self.lora_weight.size(-1) // 2), groups=self.groups)
        else:
            qwt_out = F.conv2d(x.half(), self.lora_weight.half(), None, stride=stride, padding=int(self.lora_weight.size(-1) // 2), groups=self.groups)
            qwt_out = qwt_out.float() + self.lora_bias.reshape(1, -1, 1, 1)

        out = out + qwt_out

        return out

def enable_quant(submodel):
    for name, module in submodel.named_modules():
        if isinstance(module, QuantConv2d) or isinstance(module, QuantLinear) or isinstance(module, QuantMatMul):
            module.set_quant_state(True, True)

def disable_quant(submodel):
    for name, module in submodel.named_modules():
        if isinstance(module, QuantConv2d) or isinstance(module, QuantLinear) or isinstance(module, QuantMatMul):
            module.set_quant_state(False, False)

class FeatureDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item]

def lienar_regression(X, Y, kernel_size=3, groups=4, block_id=0):
    X = gather_tensor_from_multi_processes(X, args.world_size)
    Y = gather_tensor_from_multi_processes(Y, args.world_size)

    B, C_X, H_X, W_X = X.size()
    _, C_Y, H_Y, W_Y = Y.size()

    if (H_X == H_Y) and (W_X == W_Y):
        stride = 1
    elif (H_X // 2 == H_Y) and (W_X // 2 == W_Y):
        stride = 2
    else:
        raise NotImplementedError

    # calculate channles per group
    C_per_group = C_X // groups
    _C_per_group = C_Y // groups

    # use Unfold to extract local patchs for each group
    unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=int(kernel_size//2))

    # process input and output in a group-wise manner
    weights_list = []
    bias_list = []
    for g in range(groups):
        # input channels for current group
        X_group = X[:, g * C_per_group: (g + 1) * C_per_group, :, :]

        # input patchs for current group
        X_unfold_group = unfold(X_group)  # [B, C_per_group * kernel_size * kernel_size, L]
        L = X_unfold_group.shape[-1]

        # output channels for current group
        Y_group = Y[:, g * _C_per_group: (g + 1) * _C_per_group, :, :]

        # flatting Y
        Y_flat_group = Y_group.view(B, _C_per_group, -1)  # [B, _C_per_group, L]

        # concate all batchs to form an integrated equations
        X_batch_all = X_unfold_group.permute(0, 2, 1).reshape(-1, C_per_group * kernel_size * kernel_size)  # [B*L, C_per_group * kernel_size * kernel_size]
        Y_batch_all = Y_flat_group.permute(0, 2, 1).reshape(-1, _C_per_group)  # [B*L, _C_per_group]

        # bias term
        X_with_bias = torch.cat([X_batch_all, torch.ones(X_batch_all.shape[0], 1).cuda()], dim=1)  # [B*L, C_per_group * kernel_size * kernel_size + 1]

        regularization = 1e-3
        # add regularization term in case that XTX is inreversible
        XTX = X_with_bias.T @ X_with_bias
        XTX_reg = XTX + regularization * torch.eye(XTX.shape[0]).cuda()

        # analytical solution for linear regression
        W = torch.inverse(XTX_reg) @ X_with_bias.T @ Y_batch_all  # [C_per_group * kernel_size * kernel_size + 1, _C_per_group]

        # decoule W and b
        M_group = W[:-1, :].T
        b_group = W[-1, :]

        weights_list.append(M_group)
        bias_list.append(b_group)

    M_reshaped = torch.cat(weights_list, dim=0).view(C_Y, C_X // groups, kernel_size, kernel_size)

    b_final = torch.cat(bias_list, dim=0)  # [_C]

    W = M_reshaped
    b = b_final

    Y_pred = F.conv2d(X, W, b, stride=stride, padding=kernel_size//2, groups=groups)

    abs_loss = (Y - Y_pred).abs().mean()

    ss_tot = torch.sum((Y - Y.mean(dim=0)).pow(2))
    ss_res = torch.sum((Y - Y_pred).pow(2))
    r2_score = 1 - ss_res / ss_tot

    _write('block : {}      abs : {:.6f}      r2 : {:.3f}'.format(block_id, abs_loss, r2_score))

    return W, b, r2_score

@torch.no_grad()
def generate_compensation_model(q_model, train_loader, args):
    _write('start to generate compensation model')

    torch.cuda.synchronize()
    output_t = torch.zeros(size=[0,], device=args.device)
    for i, (image, _) in tqdm(enumerate(train_loader)):
        image = image.cuda()
        t_out = q_model.forward_before_blocks(image)
        output_t = torch.cat([output_t, t_out.detach()], dim=0)
        torch.cuda.synchronize()
        if i >= (LINEAR_COMPENSATION_SAMPLES // args.batch_size // args.world_size - 1):
            break


    feature_set = FeatureDataset(output_t.detach().cpu())
    feature_loader = torch.utils.data.DataLoader(feature_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    output_previous = output_t
    sup_layers = [q_model.layer1, q_model.layer2, q_model.layer3, q_model.layer4]
    for sup_id in range(len(sup_layers)):
        current_sup_layer = sup_layers[sup_id]
        for layer_id in range(len(current_sup_layer)):

            feature_set.X = output_previous.detach().cpu()

            layer = current_sup_layer[layer_id]
            output_full_precision = torch.zeros(size=[0, ], device=args.device)
            output_quant = torch.zeros(size=[0, ], device=args.device)
            output_t_ = torch.zeros(size=[0, ], device=args.device)
            for i, t_out in tqdm(enumerate(feature_loader)):
                t_out = t_out.cuda()
                disable_quant(layer)
                full_precision_out = layer(t_out)

                enable_quant(layer)
                quant_out = layer(t_out)

                output_t_ = torch.cat([output_t_, t_out.detach()], dim=0)
                output_full_precision = torch.cat([output_full_precision, full_precision_out.detach()], dim=0)
                output_quant = torch.cat([output_quant, quant_out.detach()], dim=0)

                torch.cuda.synchronize()
                if i >= (LINEAR_COMPENSATION_SAMPLES // args.batch_size  // args.world_size - 1):
                    break

            assert torch.sum((output_previous - output_t_).abs()) < 1e-3
            global_layer_id = sum(q_model.depths[:sup_id]) + layer_id
            W, b, r2_score = lienar_regression(output_t_, output_full_precision - output_quant, kernel_size=args.kernel_size, groups=max(output_t_.size(1) // args.factor, 1), block_id=global_layer_id)
            current_sup_layer[layer_id] = CompensationBlock(W=W, b=b, r2_score=r2_score, block=current_sup_layer[layer_id], groups=max(output_t_.size(1) // args.factor, 1), linear_init=True if (global_layer_id >= args.start_block) else False, local_rank=args.local_rank, block_id=global_layer_id)
            q_model.cuda()

            qwerty_layer = current_sup_layer[layer_id]

            output_previous = torch.zeros(size=[0, ], device=args.device)
            for i, t_out in tqdm(enumerate(feature_loader)):
                t_out = t_out.cuda()
                enable_quant(qwerty_layer)
                previous_out = qwerty_layer(t_out)

                output_previous = torch.cat([output_previous, previous_out.detach()], dim=0)

                torch.cuda.synchronize()
                if i >= (LINEAR_COMPENSATION_SAMPLES // args.batch_size // args.world_size - 1):
                    break

    return q_model


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="resnet18", choices=['resnet18', 'resnet50', 'resnet101'], help="model")
parser.add_argument('--data_dir', default='/opt/Dataset/ImageNet', type=str)

parser.add_argument('--w_bits', default=4, type=int, help='bit-precision of weights')
parser.add_argument('--a_bits', default=4, type=int, help='bit-precision of activations')
parser.add_argument('--start_block', default=0, type=int)
parser.add_argument('--kernel_size', default=1, type=int)
parser.add_argument('--factor', default=64, type=int)

parser.add_argument("--batch_size", default=32, type=int, help="batchsize of validation set")
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument("--seed", default=0, type=int, help="seed")

parser.add_argument("--local-rank", default=0, type=int)
args = parser.parse_args()

train_aug = 'large_scale_train'
test_aug = 'large_scale_test'
args.drop_path = 0.0
args.num_classes = 1000

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
crop_pct = 0.875

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1
args.device = 'cuda:0'
args.world_size = 1
args.rank = 0  # global rank
if args.distributed:
    args.device = 'cuda:%d' % args.local_rank
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()
    args.rank = torch.distributed.get_rank()

assert args.rank >= 0


args.log_dir = os.path.join('checkpoint', args.model, 'QwTGroupConv', 'bs_{}_worldsize_{}_w_{}_a_{}_kernelsize_{}_factor_{}_startblock_{}_sed_{}' .format(args.batch_size, args.world_size, args.w_bits, args.a_bits, args.kernel_size, args.factor, args.start_block, args.seed))

args.log_file = os.path.join(args.log_dir, 'log.txt')


if args.local_rank == 0:
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if os.path.isfile(args.log_file):
        os.remove(args.log_file)
else:
    time.sleep(1)

torch.cuda.synchronize()

_write = partial(write, log_file=args.log_file)

if args.distributed:
    _write('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.' % (args.rank, args.world_size))
else:
    _write('Training with a single process on 1 GPUs.')
assert args.rank >= 0


def main():

    if args.local_rank == 0:
        _write(args)

    seed(args.seed)

    if args.local_rank == 0:
        _write('dataset mean : {} & std : {}'.format(mean, std))

    dataset_train = ImageDataset(root=os.path.join(args.data_dir, 'train'), transform=create_transform(train_aug, mean, std, crop_pct))
    dataset_eval = ImageDataset(root=os.path.join(args.data_dir, 'val'), transform=create_transform(test_aug, mean, std, crop_pct))

    if args.local_rank == 0:
        _write('len of train_set : {}    train_transform : {}'.format(len(dataset_train), dataset_train.transform))
        _write('len of eval_set : {}    eval_transform : {}'.format(len(dataset_eval), dataset_eval.transform))


    loader_train = create_loader(
        dataset_train,
        batch_size=args.batch_size,
        is_training=True,
        re_prob=0.0,
        mean=mean,
        std=std,
        num_workers=args.num_workers,
        distributed=args.distributed,
        log_file=args.log_file,
        drop_last=True,
        local_rank=args.local_rank,
        persistent_workers=False
    )

    loader_eval = create_loader(
        dataset_eval,
        batch_size=args.batch_size,
        is_training=False,
        re_prob=0.,
        mean=mean,
        std=std,
        num_workers=args.num_workers,
        distributed=args.distributed,
        log_file=args.log_file,
        drop_last=False,
        local_rank=args.local_rank,
        persistent_workers=False
    )

    for data, _ in loader_train:
        calib_data = data.to(args.device)
        break

    broadcast_tensor_from_main_process(calib_data, args)
    _write('local_rank : {} calib_data shape : {} value : {}'.format(args.local_rank, calib_data.size(), calib_data[0, 0, 0, :5]))



    _write('Building model ...')
    if args.model == 'resnet18':
        model = resnet18(num_classes=args.num_classes, pretrained=False)
    elif args.model == 'resnet50':
        model = resnet50(num_classes=args.num_classes, pretrained=False)
    elif args.model == 'resnet101':
        model = resnet101(num_classes=args.num_classes, pretrained=False)
    else:
        raise NotImplementedError

    checkpoint = torch.load(model_path[args.model], map_location='cpu')
    model.load_state_dict(checkpoint)

    model.to(args.device)
    model.eval()

    fp32_model = copy.deepcopy(model)
    base_model = fp32_model

    wq_params = {'n_bits': args.w_bits, 'channel_wise': True}
    aq_params = {'n_bits': args.a_bits, 'channel_wise': False}
    q_model = quant_model_resnet(model, input_quant_params=aq_params, weight_quant_params=wq_params)
    q_model.to(args.device)
    q_model.eval()

    # Initial quantization
    _write('Performing initial quantization ...')
    set_quant_state(q_model, input_quant=True, weight_quant=True)

    # for resnet, only the percentile-calibration is performed
    with torch.no_grad():
        _ = q_model(calib_data)


    fp32_params = compute_quantized_params(fp32_model, local_rank=args.local_rank, log_file=args.log_file)
    _write('FP32 model size is {:.3f}'.format(fp32_params))

    ptq_params = compute_quantized_params(q_model, local_rank=args.local_rank, log_file=args.log_file)
    _write('Percentile model size is {:.3f}'.format(ptq_params))

    top1_acc_eval = validate(fp32_model, loader_eval)
    _write('FP32 model   eval_acc: {:.2f}'.format(top1_acc_eval.avg))

    top1_acc_eval = validate(q_model, loader_eval)
    _write('Percentile   eval_acc: {:.2f}'.format(top1_acc_eval.avg))

    q_model = generate_compensation_model(q_model, loader_train, args)

    qwerty_params = compute_quantized_params(q_model, local_rank=args.local_rank, log_file=args.log_file)
    _write('QwT model size is {:.3f}'.format(qwerty_params))

    for name, module in q_model.named_modules():
        if isinstance(module, QuantConv2d) or isinstance(module, QuantLinear) or isinstance(module, QuantMatMul):
            if hasattr(module, 'use_input_quant'):
                use_input_quant = module.use_input_quant
            else:
                use_input_quant = None

            if hasattr(module, 'use_weight_quant'):
                use_weight_quant = module.use_weight_quant
            else:
                use_weight_quant = None

            if args.local_rank == 0:
                _write('module : {}, input_quant : {}, weight_quant : {}'.format(name, use_input_quant, use_weight_quant))


    top1_acc_eval = validate(q_model, loader_eval)
    _write('Percentile + QwT   eval_acc: {:.2f}'.format(top1_acc_eval.avg))

def validate(model, loader):
    top1_m = AverageMeter()

    model.eval()

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):

            input = input.cuda()
            target = target.cuda()

            _, output = model(input)

            acc1, _ = accuracy(output, target, topk=(1, 5))

            top1_m.update(acc1.item(), output.size(0))

        top1_m.synchronize()

    _write('Test  Smples : {top1.count}    Acc@1: {top1.avg:>7.4f}'.format(top1=top1_m))
    return top1_m


if __name__ == '__main__':
    main()
