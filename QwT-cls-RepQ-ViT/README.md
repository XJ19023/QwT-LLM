## Quantization without Tears

The following instructions explain how to reproduce QwT’s ImageNet classification results using the post-training quantization method [RepQ-ViT](https://arxiv.org/abs/2212.08254).


## Evaluation

- You can quantize and evaluate model using the following command:

```bash
#single GPU
python qwt_vit_and_deit.py [--model] [--data_dir] [--w_bit] [--a_bit]

#multi-GPU
CUDA_VISIBLE_DEVICE=<YOUR GPU ID> python -m torch.distributed.launch --nproc_per_node <YOUR GPU NUMS> --master_port <YOUR PORT> qwt_vit_and_deit.py [--model] [--w_bit] [--a_bit]

scripts:
   on vit & deit --- qwt_vit_and_deit.py
   on swin       --- qwt_swin.py
optional arguments:
--model: Model architecture, the choises can be: 
    vit_small, vit_base, deit_tiny, deit_small, deit_base, swin_tiny, swin_small ...
--data_dir: Path to ImageNet dataset.
--w_bit: Bit-precision of weights, default=4.
--a_bit: Bit-precision of activation, default=4.
```

- Example: Quantize *DeiT-S* at W4/A4 precision:

```bash
# single GPU
CUDA_VISIBLE_DEVICES=0 python qwt_vit_and_deit.py --model deit_small --data_dir <YOUR_DATA_DIR> --w_bit 4 --a_bit 4

# 4 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12661 python qwt_vit_and_deit.py --model deit_small --data_dir <YOUR_DATA_DIR> --w_bit 4 --a_bit 4
```

## Results

Below are the results obtained on the ImageNet dataset using our proposed QwT.

| Model                    | Method         | Prec. | Top-1(%) | Prec. | Top-1(%) |
|:--------------:          |:-------------: |:-----:|:--------:|:-----:|:--------:|
| DeiT-T (72.2)            |  RepQ-ViT      | W4/A4 |  58.2    | W6/A6 |  71.0   |
|                          | RepQ-ViT + QwT | W4/A4 |  61.4    | W6/A6 |  71.2   |
| DeiT-S (79.9)            |  RepQ-ViT      | W4/A4 |  69.0    | W6/A6 |  78.9   |
|                          | RepQ-ViT + QwT | W4/A4 |  71.5    | W6/A6 |  79.1   |
| DeiT-B (81.8)            |  RepQ-ViT      | W4/A4 |  75.9    | W6/A6 |  81.2   |
|                          | RepQ-ViT + QwT | W4/A4 |  77.1    | W6/A6 |  81.4   |
| DeiT-T-distilled (74.5)  |  RepQ-ViT      | W4/A4 |  62.9    | W6/A6 |  73.3   |
|                          | RepQ-ViT + QwT | W4/A4 |  64.7    | W6/A6 |  73.6   |
| DeiT-S-distilled (81.2)  |  RepQ-ViT      | W4/A4 |  70.2    | W6/A6 |  80.4   |
|                          | RepQ-ViT + QwT | W4/A4 |  76.2    | W6/A6 |  80.5   |
| DeiT-B-distilled (83.3)  |  RepQ-ViT      | W4/A4 |  75.7    | W6/A6 |  82.7   |
|                          | RepQ-ViT + QwT | W4/A4 |  77.7    | W6/A6 |  82.8   |
| ViT-S (81.4)             |  RepQ-ViT      | W4/A4 |  65.8    | W6/A6 |  80.5   |
|                          | RepQ-ViT + QwT | W4/A4 |  70.8    | W6/A6 |  80.7   |
| ViT-B (84.5)             |  RepQ-ViT      | W4/A4 |  68.5    | W6/A6 |  83.6   |
|                          | RepQ-ViT + QwT | W4/A4 |  76.3    | W6/A6 |  83.9   |
| Swin-T (81.4)            |  RepQ-ViT      | W4/A4 | 73.0     | W6/A6 | 80.6    |
|                          | RepQ-ViT + QwT | W4/A4 | 75.5     | W6/A6 | 80.7    |
| Swin-S (83.2)            |  RepQ-ViT      | W4/A4 | 80.2     | W6/A6 | 82.8    |
|                          | RepQ-ViT + QwT | W4/A4 | 80.4     | W6/A6 | 82.9    |

## Citation

We would greatly appreciate it if you could cite our paper if you find our implementation helpful in your work.

```bash
@InProceedings{Fu_2025_CVPR,
    author    = {Fu, Minghao and Yu, Hao and Shao, Jie and Zhou, Junjie and Zhu, Ke and Wu, Jianxin},
    title     = {Quantization without Tears},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {4462-4472}
}
```
