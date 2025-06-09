<div align=center>
  <img src="imgs/QwT_illustration.png" width="500px" />
</div>


# Quantization without Tears (CVPR 2025)

Implementation of **QwT**, a simple, fast, and general approach to network quantization that “adds no tears” to your workflow. QwT augments any PTQ model with lightweight linear compensation layers to recover information lost during quantization .

---

## 📖 Paper

> **Minghao Fu**, Hao Yu, Jie Shao, Junjie Zhou, Ke Zhu & Jianxin Wu  
> **Quantization without Tears**, the Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)  
> [PDF](https://arxiv.org/pdf/2411.13918.pdf) • [CVPR 2025 Version](https://openaccess.thecvf.com/content/CVPR2025/html/Fu_Quantization_without_Tears_CVPR_2025_paper.html)

---


## 🚀 Features

- **Fast**: closed‐form compensation parameters can be computed on a small calibration set in under 2 minutes.  
- **Accurate**: outperforms standard PTQ methods without any back‐prop.  
- **Simple**: zero task‐specific hyperparameters; only a few linear layers (`W, b`) per block.  
- **General**: applies to CNNs (ResNet), Transformers (ViT, Swin), detection (Mask R-CNN, DETR), segmentation, multi‐modal (CLIP), generative (DiT) and LLMs (LLaMA).  
- **Practical**: integrates seamlessly with TensorRT or any existing INT8/PTQ pipeline.

---

## 📦 Installation

- We recommend using Timm version 0.4.12.

- **To install QwT and develop locally:**

```bash
  git clone https://github.com/wujx2001/qwt.git
  cd qwt
```

## 🛠️ Usage

For detailed reproduction instructions, please refer to:
- [**QwT-Classification-RepQ-ViT README**](https://github.com/wujx2001/QwT/blob/main/QwT-cls-RepQ-ViT/README.md) — how to reproduce ImageNet classification results of QwT using RepQ-ViT as baseline.
- [**QwT-Detection-RepQ-ViT README**](https://github.com/wujx2001/QwT/blob/main/QwT-det-RepQ-ViT/README.md) — how to reproduce COCO detection results of QwT using RepQ-ViT as baseline.

---

## 🙏 Acknowledgements

This implementation builds on code from [RepQ-ViT](https://github.com/zkkli/RepQ-ViT) and leverages the [timm](https://github.com/rwightman/pytorch-image-models) library.


## 🎓 Citation

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