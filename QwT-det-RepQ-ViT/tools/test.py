import argparse
import os
import warnings

import mmcv
import torch
from tqdm import tqdm
import torch.distributed
from torch.utils.data import Dataset
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
import torch.distributed as dist

from quant import *

LINEAR_COMPENSATION_SAMPLES = 512

def gather_tensor_from_multi_processes(input, world_size):
    if world_size == 1:
        return input
    torch.cuda.synchronize()
    gathered_tensors = [torch.zeros_like(input) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, input)
    gathered_tensors = torch.cat(gathered_tensors, dim=0)
    torch.cuda.synchronize()

    return gathered_tensors

def broadcast_tensor_from_main_process(input, args):
    if not args.distributed:
        return
    torch.cuda.synchronize()
    src_rank = 0
    dist.broadcast(input, src=src_rank)
    torch.cuda.synchronize()
    return input

def broadcast_object_list_from_main_process(input, args):
    if not args.distributed:
        return
    torch.cuda.synchronize()
    data_list = [input]
    dist.broadcast_object_list(data_list, src=0)
    torch.cuda.synchronize()
    return data_list[0]

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

class CompensationBlock(nn.Module):
    def __init__(self, W, b, r2_score, block, linear_init=True, local_rank=0, block_id=None):
        super(CompensationBlock, self).__init__()
        self.block = block

        self.lora_weight = nn.Parameter(torch.zeros((W.size(0), W.size(1))))
        self.lora_bias = nn.Parameter(torch.zeros(W.size(1)))

        if linear_init and (r2_score > 0):
            self.lora_weight.data.copy_(W)
            self.lora_bias.data.copy_(b)
            if local_rank == 0:
                print('block {} using linear init'.format(block_id))
        else:
            nn.init.zeros_(self.lora_weight)
            nn.init.zeros_(self.lora_bias)
            if local_rank == 0:
                print('block {} using lora init'.format(block_id))

    def forward(self, x, mask_matrix):
        out = self.block(x, mask_matrix)

        out = out + x @ self.lora_weight + self.lora_bias

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

def lienar_regression(X, Y, block_id=0):
    X = X.reshape(-1, X.size(-1))

    X = gather_tensor_from_multi_processes(X, world_size=args.world_size)

    X_add_one = torch.cat([X, torch.ones(size=[X.size(0), ], device=X.device).reshape(-1, 1)], dim=-1)
    Y = Y.reshape(-1, Y.size(-1))

    Y = gather_tensor_from_multi_processes(Y, world_size=args.world_size)

    print('the shape of X_add_one is {}, Y is {}'.format(X_add_one.size(), Y.size()))

    X_add_one_T = X_add_one.t()
    W_overall = torch.inverse(X_add_one_T @ X_add_one) @ X_add_one_T @ Y

    W = W_overall[:-1, :]
    b = W_overall[-1, :]

    Y_pred = X @ W + b

    abs_loss = (Y - Y_pred).abs().mean()

    ss_tot = torch.sum((Y - Y.mean(dim=0)).pow(2))
    ss_res = torch.sum((Y - Y_pred).pow(2))
    r2_score = 1 - ss_res / ss_tot

    print('block : {}      abs : {:.6f}      r2 : {:.3f}'.format(block_id, abs_loss, r2_score))

    return W, b, r2_score

@torch.no_grad()
def generate_compensation_model(q_model, train_loader, args):
    print('start to generate compensation model')


    torch.cuda.synchronize()
    output_t = torch.zeros(size=[0,], device=args.device)
    for i, image_metas in tqdm(enumerate(train_loader)):
        image = image_metas['img'][0].cuda()
        t_out, H, W = q_model.forward_before_blocks(image)
        output_t = torch.cat([output_t, t_out.detach()], dim=0)
        torch.cuda.synchronize()
        if i >= (LINEAR_COMPENSATION_SAMPLES // args.batch_size // args.world_size - 1):
            break


    feature_set = FeatureDataset(output_t.detach().cpu())
    feature_loader = torch.utils.data.DataLoader(feature_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    output_previous = output_t
    for layer_id in range(len(q_model.layers)):
        current_layer = q_model.layers[layer_id]
        Hp = int(np.ceil(H / current_layer.window_size)) * current_layer.window_size
        Wp = int(np.ceil(W / current_layer.window_size)) * current_layer.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=args.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -current_layer.window_size), slice(-current_layer.window_size, -current_layer.shift_size), slice(-current_layer.shift_size, None))
        w_slices = (slice(0, -current_layer.window_size), slice(-current_layer.window_size, -current_layer.shift_size), slice(-current_layer.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, current_layer.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, current_layer.window_size * current_layer.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for block_id in range(len(current_layer.blocks)):

            feature_set.X = output_previous.detach().cpu()

            block = current_layer.blocks[block_id]
            block.H, block.W = H, W

            output_full_precision = torch.zeros(size=[0, ], device=args.device)
            output_quant = torch.zeros(size=[0, ], device=args.device)
            output_t_ = torch.zeros(size=[0, ], device=args.device)
            for i, t_out in tqdm(enumerate(feature_loader)):
                t_out = t_out.cuda()
                disable_quant(block)
                full_precision_out = block(t_out, attn_mask)

                enable_quant(block)
                quant_out = block(t_out, attn_mask)

                output_t_ = torch.cat([output_t_, t_out.detach()], dim=0)
                output_full_precision = torch.cat([output_full_precision, full_precision_out.detach()], dim=0)
                output_quant = torch.cat([output_quant, quant_out.detach()], dim=0)

                torch.cuda.synchronize()
                if i >= (LINEAR_COMPENSATION_SAMPLES // args.batch_size  // args.world_size - 1):
                    break

            assert torch.sum((output_previous - output_t_).abs()) < 1e-3
            global_block_id = sum(q_model.depths[:layer_id]) + block_id
            _W, b, r2_score = lienar_regression(output_t_, output_full_precision - output_quant, block_id=global_block_id)
            current_layer.blocks[block_id] = CompensationBlock(W=_W, b=b, r2_score=r2_score, block=current_layer.blocks[block_id], linear_init=True if global_block_id >= args.start_block else False, local_rank=args.local_rank, block_id=global_block_id)
            q_model.cuda()

            qwerty_block = current_layer.blocks[block_id]
            qwerty_block.H, qwerty_block.W = H, W

            output_previous = torch.zeros(size=[0, ], device=args.device)
            for i, t_out in tqdm(enumerate(feature_loader)):
                t_out = t_out.cuda()
                enable_quant(qwerty_block)
                previous_out = qwerty_block(t_out, attn_mask)

                if (current_layer.downsample is not None) and (block_id == len(current_layer.blocks)-1):
                    previous_out = current_layer.downsample(previous_out, H, W)

                output_previous = torch.cat([output_previous, previous_out.detach()], dim=0)

                torch.cuda.synchronize()
                if i >= (LINEAR_COMPENSATION_SAMPLES // args.batch_size // args.world_size - 1):
                    break
        H, W = (H + 1) // 2, (W + 1) // 2


    return q_model

parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
parser.add_argument('config', help='test config file path')
parser.add_argument('checkpoint', help='checkpoint file')
parser.add_argument('--out', help='output result file in pickle format')
parser.add_argument(
    '--fuse-conv-bn',
    action='store_true',
    help='Whether to fuse conv and bn, this will slightly increase'
    'the inference speed')
parser.add_argument(
    '--format-only',
    action='store_true',
    help='Format the output results without perform evaluation. It is'
    'useful when you want to format the result to a specific format and '
    'submit it to the test server')
parser.add_argument(
    '--eval',
    type=str,
    nargs='+',
    default=['bbox', 'segm'],
    help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
    ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
parser.add_argument('--show', action='store_true', help='show results')
parser.add_argument(
    '--show-dir', help='directory where painted images will be saved')
parser.add_argument(
    '--show-score-thr',
    type=float,
    default=0.3,
    help='score threshold (default: 0.3)')
parser.add_argument(
    '--gpu-collect',
    action='store_true',
    help='whether to use gpu to collect results.')
parser.add_argument(
    '--tmpdir',
    help='tmp directory used for collecting results from multiple '
    'workers, available when gpu-collect is not specified')
parser.add_argument(
    '--cfg-options',
    nargs='+',
    action=DictAction,
    help='override some settings in the used config, the key-value pair '
    'in xxx=yyy format will be merged into config file. If the value to '
    'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
    'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
    'Note that the quotation marks are necessary and that no white space '
    'is allowed.')
parser.add_argument(
    '--options',
    nargs='+',
    action=DictAction,
    help='custom options for evaluation, the key-value pair in xxx=yyy '
    'format will be kwargs for dataset.evaluate() function (deprecate), '
    'change to --eval-options instead.')
parser.add_argument(
    '--eval-options',
    nargs='+',
    action=DictAction,
    help='custom options for evaluation, the key-value pair in xxx=yyy '
    'format will be kwargs for dataset.evaluate() function')
parser.add_argument(
    '--launcher',
    choices=['none', 'pytorch', 'slurm', 'mpi'],
    default='none',
    help='job launcher')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--w_bits', default=4,
                    type=int, help='bit-precision of weights')
parser.add_argument('--a_bits', default=4,
                    type=int, help='bit-precision of activation')

args = parser.parse_args()
if 'LOCAL_RANK' not in os.environ:
    os.environ['LOCAL_RANK'] = str(args.local_rank)

if args.options and args.eval_options:
    raise ValueError(
        '--options and --eval-options cannot be both '
        'specified, --options is deprecated in favor of --eval-options')
if args.options:
    warnings.warn('--options is deprecated in favor of --eval-options')
    args.eval_options = args.options


def main():

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        args.world_size = 1
        args.device = 'cuda:0'
        args.distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        args.world_size = torch.distributed.get_world_size()
        args.distributed = True

    # build the dataloader
    test_dataset = build_dataset(cfg.data.test)
    test_loader = build_dataloader(
        test_dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    qwerty_calibration_set = build_dataset(cfg.data.qwerty_calibration)
    qwerty_calibration_loader = build_dataloader(
            qwerty_calibration_set,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=True
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = test_dataset.CLASSES


    # build the quantized model
    wq_params = {'n_bits': args.w_bits, 'channel_wise': True}
    aq_params = {'n_bits': args.a_bits, 'channel_wise': False}
    q_model = quant_model(model, input_quant_params=aq_params, weight_quant_params=wq_params)
    q_model.cuda()
    q_model.eval()

    if not distributed:
        q_model = MMDataParallel(q_model, device_ids=[0])
        dataset = test_loader.dataset

    else:
        q_model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        dataset = test_loader.dataset

    # initial quantization
    print('performing initial quantization ...')
    set_quant_state(q_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        for i, calib_data in enumerate(test_loader):
            print('local_rank : {}   calib_data_before : {}'.format(args.local_rank, calib_data['img_metas'][0].data[0][0]['filename']))

            calib_data = broadcast_object_list_from_main_process(calib_data, args)
            result = q_model(return_loss=False, rescale=True, **calib_data)  # (1, 3, 800, 1216)
            break

    print('local_rank : {}   calib_data : {}'.format(args.local_rank, calib_data['img_metas'][0].data[0][0]['filename']))
    torch.cuda.synchronize()

    # scale reparameterization
    print('performing scale reparameterization ...')
    with torch.no_grad():
        module_dict = {}
        for name, module in q_model.module.backbone.layers.named_modules():
            module_dict[name] = module
            idx = name.rfind('.')
            if idx == -1:
                idx = 0
            father_name = name[:idx]
            if father_name in module_dict:
                father_module = module_dict[father_name]
            else:
                raise RuntimeError(f"father module {father_name} not found")

            if 'norm1' in name or 'norm2' in name or 'norm' in name:
                # next_module = father_module.attn.qkv if 'norm1' in name else father_module.mlp.fc1
                if 'norm1' in name:
                    next_module = father_module.attn.qkv
                elif 'norm2' in name:
                    next_module = father_module.mlp.fc1
                else:
                    next_module = father_module.reduction

                act_delta = next_module.input_quantizer.delta.reshape(-1)
                act_zero_point = next_module.input_quantizer.zero_point.reshape(-1)
                act_min = -act_zero_point * act_delta

                target_delta = torch.mean(act_delta)
                target_zero_point = torch.mean(act_zero_point)
                target_min = -target_zero_point * target_delta

                r = act_delta / target_delta
                b = act_min / r - target_min

                module.weight.data = module.weight.data / r
                module.bias.data = module.bias.data / r - b

                next_module.weight.data = next_module.weight.data * r
                if next_module.bias is not None:
                    next_module.bias.data = next_module.bias.data + torch.mm(next_module.weight.data,
                                                                             b.reshape(-1, 1)).reshape(-1)
                else:
                    next_module.bias = Parameter(torch.Tensor(next_module.out_features))
                    next_module.bias.data = torch.mm(next_module.weight.data, b.reshape(-1, 1)).reshape(-1)

                next_module.input_quantizer.channel_wise = False
                next_module.input_quantizer.delta = target_delta
                next_module.input_quantizer.zero_point = target_zero_point
                next_module.weight_quantizer.inited = False

    # re-calibration
    set_quant_state(q_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        result = q_model(return_loss=False, rescale=True, **calib_data)


    args.batch_size = cfg.data.samples_per_gpu
    args.num_workers = cfg.data.workers_per_gpu
    args.start_block = 0


    if not distributed:
        outputs = single_gpu_test(q_model, test_loader, args.show, args.show_dir, args.show_score_thr)

    else:
        outputs = multi_gpu_test(q_model, test_loader, args.tmpdir, args.gpu_collect)

    print('Now we generate the evaluation results using RepQ-ViT...')
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))

    q_model.module.backbone = generate_compensation_model(q_model.module.backbone, qwerty_calibration_loader, args)

    if not distributed:
        outputs = single_gpu_test(q_model, test_loader, args.show, args.show_dir, args.show_score_thr)

    else:
        outputs = multi_gpu_test(q_model, test_loader, args.tmpdir, args.gpu_collect)

    print('Now we generate the evaluation results using RepQ-ViT + QwT...')
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
