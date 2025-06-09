# Quantization without Tears

This repository extends [mmdetection](https://github.com/open-mmlab/mmdetection) and [RepQ-ViT-Detection](https://github.com/zkkli/RepQ-ViT/tree/main/detection) to reproduce QwT’s object-detection results on COCO.

---

## Preliminaries

- **Install MMCV via MIM**  
  ```bash
  pip install -U openmim
  mim install mmcv-full
  ```

- **Install MMDetection**
    ```bash
    cd QwT/detection
    pip install -v -e .
    ```

- Get the checkpoints from the [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) repo and place them in your *pretrained_weights* directory.

- Update the **data_root** setting in `configs/_base_/datasets/coco_instance.py` to point to the root directory of your COCO dataset.


## Evaluation

- You can quantize and evaluate model using the following command:

```bash
CUDA_VISIBLE_DEVICES=<YOUR GPU IDs> tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <\#GPU USED> --eval bbox segm [--w_bits] [--a_bits]

Required arguments:
 <CONFIG_FILE> : Path to config. You can find it at ./configs/swin/
 <DET_CHECKPOINT_FILE> : Path to checkpoint of pre-trained models.

optional arguments:
--w_bit: Bit-precision of weights, default=4.
--a_bit: Bit-precision of activation, default=4.
```

- Example: Quantize *Cascade Mask R-CNN with Swin-T* at W4/A4 precision:

```bash
# single GPUs
CUDA_VISIBLE_DEVICES=0 bash tools/dist_test.sh configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py pretrained_weights/cascade_mask_rcnn_swin_tiny_patch4_window7.pth 1  --eval bbox segm --w_bit 4 --a_bits 4

# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_test.sh configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py pretrained_weights/cascade_mask_rcnn_swin_tiny_patch4_window7.pth 4  --eval bbox segm --w_bit 4 --a_bits 4
```

### Results

Below are the experimental results obtained on the COCO dataset using QwT.

| Model                                     | Method           | Prec. | AP<sup>box</sup> / AP<sup>mask</sup> | Prec. | AP<sup>box</sup> / AP<sup>mask</sup> |
|:-----------------------------------------:|:----------------:|:-----:|:------------------------------------:|:-----:|:------------------------------------:|
| Mask RCNN + Swin_T (46.0 / 41.6)          |  RepQ-ViT        | W4/A4 | 36.1 / 36.0                          | W6/A6 | 45.5 / 41.3                          |  
|                                           |  RepQ-ViT + QwT  | W4/A4 | 36.3 / 36.0                          | W6/A6 | 45.4 / 41.3                          |
| Mask RCNN + Swin_S (48.5 / 43.3)          |  RepQ-ViT        | W4/A4 | 42.6 / 40.0                          | W6/A6 | 47.6 / 42.9                          |
|                                           |  RepQ-ViT + QwT  | W4/A4 | 43.1 / 40.4                          | W6/A6 | 48.0 / 43.1                          |
| Cascade Mask RCNN + Swin_T (50.4 / 43.7)  |  RepQ-ViT        | W4/A4 | 47.0 / 41.4                          | W6/A6 | 50.1 / 43.5                          |
|                                           |  RepQ-ViT + QwT  | W4/A4 | 47.6 / 41.8                          | W6/A6 | 50.1 / 43.6                          |
| Cascade Mask RCNN + Swin_S (51.9 / 45.0)  |  RepQ-ViT        | W4/A4 | 49.3 / 43.1                          | W6/A6 | 51.4 / 44.6                          |
|                                           |  RepQ-ViT + QwT  | W4/A4 | 49.9 / 43.4                          | W6/A6 | 51.7 / 44.8                          |
| Cascade Mask RCNN + Swin_B (51.9 / 45.0)  |  RepQ-ViT        | W4/A4 | 49.3 / 43.1                          | W6/A6 | 51.5 / 44.8                          |
|                                           |  RepQ-ViT + QwT  | W4/A4 | 50.0 / 43.7                          | W6/A6 | 51.8 / 45.0                          |

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
