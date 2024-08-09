# Compact and Mighty - Image Tokenization with Only 32 Tokens for both Reconstruction and Generation!

<div align="center">

[![demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Online_Demo-blue)](https://huggingface.co/spaces/fun-research/TiTok)&nbsp;&nbsp;
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://yucornetto.github.io/projects/titok.html)&nbsp;&nbsp;
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2406.07550)&nbsp;&nbsp;

</div>

We present a compact 1D tokenizer which can represent an image with as few as 32 discrete tokens. As a result, it leads to a substantial speed-up on the sampling process (e.g., **410 × faster** than DiT-XL/2) while obtaining a competitive generation quality.


<p>
<img src="assets/titok_teaser.png" alt="teaser" width=90% height=90%>
</p>
<p>
<img src="assets/speed_vs_perf.png" alt="teaser" width=90% height=90%>
</p>

## 🚀 Contributions

#### We introduce a novel 1D image tokenization framework that breaks grid constraints existing in 2D tokenization methods, leading to a much more flexible and compact image latent representation.

#### The proposed 1D tokenizer can tokenize a 256 × 256 image into as few as 32 discrete tokens, leading to a signigicant speed-up (hundreds times faster than diffusion models) in generation process, while maintaining state-of-the-art generation quality.

#### We conduct a series of experiments to probe the properties of rarely studied 1D image tokenization, paving the path towards compact latent space for efficient and effective image representation.

## Model Zoo
| Dataset  | Model | Link | FID |
| ------------- | ------------- | ------------- | ------------- |
| ImageNet  | TiTok-L-32 Tokenizer | [checkpoint](https://huggingface.co/fun-research/TiTok/blob/main/tokenizer_titok_l32.bin)| 2.21 (reconstruction) |
| ImageNet  | TiTok-B-64 Tokenizer | [checkpoint](https://huggingface.co/fun-research/TiTok/blob/main/tokenizer_titok_b64.bin) | 1.70 (reconstruction) |
| ImageNet  | TiTok-S-128 Tokenizer | [checkpoint](https://huggingface.co/fun-research/TiTok/blob/main/tokenizer_titok_s128.bin) | 1.71 (reconstruction) |
| ImageNet  | TiTok-L-32 Generator | [checkpoint](https://huggingface.co/fun-research/TiTok/blob/main/generator_titok_l32.bin) | 2.77 (generation) |
| ImageNet  | TiTok-B-64 Generator | [checkpoint](https://huggingface.co/fun-research/TiTok/blob/main/generator_titok_b64.bin) | 2.48 (generation) |
| ImageNet  | TiTok-S-128 Generator | [checkpoint](https://huggingface.co/fun-research/TiTok/blob/main/generator_titok_s128.bin) | 1.97 (generation) |

Please note that these models are trained only on limited academic dataset ImageNet, and they are only for research purposes.

## Installation
```shell
pip3 install -r requirements.txt
```

## Get Started

```python
import torch
from PIL import Image
import numpy as np
import demo_util
from huggingface_hub import hf_hub_download
from modeling.maskgit import ImageBert
from modeling.titok import TiTok

titok_tokenizer = TiTok.from_pretrained("yucornetto/tokenizer_titok_l32_imagenet")
titok_generator = ImageBert.from_pretrained("yucornetto/generator_titok_l32_imagenet")

# or alternatively, downloads from hf
# hf_hub_download(repo_id="fun-research/TiTok", filename="tokenizer_titok_l32.bin", local_dir="./")
# hf_hub_download(repo_id="fun-research/TiTok", filename="generator_titok_l32.bin", local_dir="./")

# load config
# config = demo_util.get_config("configs/titok_l32.yaml")
# titok_tokenizer = demo_util.get_titok_tokenizer(config)
# titok_generator = demo_util.get_titok_generator(config)

device = "cuda"
titok_tokenizer = titok_tokenizer.to(device)
titok_generator = titok_generator.to(device)

# reconstruct an image. I.e., image -> 32 tokens -> image
img_path = "assets/ILSVRC2012_val_00010240.png"
image = torch.from_numpy(np.array(Image.open(img_path)).astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
# tokenization
encoded_tokens = titok_tokenizer.encode(image.to(device))[1]["min_encoding_indices"]
# image assets/ILSVRC2012_val_00010240.png is encoded into tokens tensor([[[ 887, 3979,  349,  720, 2809, 2743, 2101,  603, 2205, 1508, 1891, 4015, 1317, 2956, 3774, 2296,  484, 2612, 3472, 2330, 3140, 3113, 1056, 3779,  654, 2360, 1901, 2908, 2169,  953, 1326, 2598]]], device='cuda:0'), with shape torch.Size([1, 1, 32])
print(f"image {img_path} is encoded into tokens {encoded_tokens}, with shape {encoded_tokens.shape}")
# de-tokenization
reconstructed_image = titok_tokenizer.decode_tokens(encoded_tokens)
reconstructed_image = torch.clamp(reconstructed_image, 0.0, 1.0)
reconstructed_image = (reconstructed_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
reconstructed_image = Image.fromarray(reconstructed_image).save("assets/ILSVRC2012_val_00010240_recon.png")

# generate an image
sample_labels = [torch.randint(0, 999, size=(1,)).item()] # random IN-1k class
generated_image = demo_util.sample_fn(
    generator=titok_generator,
    tokenizer=titok_tokenizer,
    labels=sample_labels,
    guidance_scale=4.5,
    randomize_temperature=1.0,
    num_sample_steps=8,
    device=device
)
Image.fromarray(generated_image[0]).save(f"assets/generated_{sample_labels[0]}.png")
```

We also provide a [jupyter notebook](demo.ipynb) for a quick tutorial on reconstructing and generating images with TiTok-L-32.

We also support TiTok with [HuggingFace 🤗 Demo](https://huggingface.co/spaces/fun-research/TiTok)!

## Testing on ImageNet-1K Benchmark

We provide a [sampling script](./sample_imagenet.py) for reproducing the generation results on ImageNet-1K benchmark.
```bash
# Prepare ADM evaluation script
git clone https://github.com/openai/guided-diffusion.git

wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
```
```python
# Reproducing TiTok-L-32
torchrun --nnodes=1 --nproc_per_node=8 --rdzv-endpoint=localhost:9999 sample_imagenet.py config=configs/titok_l32.yaml experiment.output_dir="titok_l_32"
# Run eval script. The result FID should be ~2.77
python3 guided-diffusion/evaluations/evaluator.py VIRTUAL_imagenet256_labeled.npz titok_l_32.npz

# Reproducing TiTok-B-64
torchrun --nnodes=1 --nproc_per_node=8 --rdzv-endpoint=localhost:9999 sample_imagenet.py config=configs/titok_b64.yaml experiment.output_dir="titok_b_64"
# Run eval script. The result FID should be ~2.48
python3 guided-diffusion/evaluations/evaluator.py VIRTUAL_imagenet256_labeled.npz titok_b_64.npz

# Reproducing TiTok-S-128
torchrun --nnodes=1 --nproc_per_node=8 --rdzv-endpoint=localhost:9999 sample_imagenet.py config=configs/titok_s128.yaml experiment.output_dir="titok_s_128"
# Run eval script. The result FID should be ~1.97
python3 guided-diffusion/evaluations/evaluator.py VIRTUAL_imagenet256_labeled.npz titok_s_128.npz
```
## Updates
- 08/09/2024: Better support on loading pretrained weights from huggingface models, thanks for the help from [@NielsRogge](https://github.com/NielsRogge)！
- 07/03/2024: Evaluation scripts for reproducing the results reported in the paper, checkpoints of TiTok-B64 and TiTok-S128 are available.
- 06/21/2024: Demo code and TiTok-L-32 checkpoints release. 
- 06/11/2024: The [tech report](https://arxiv.org/abs/2406.07550) of this project is available.

## Visualizations
<p>
<img src="assets/recon_w_model_size_num_token.png" alt="teaser" width=90% height=90%>
</p>
<p>
<img src="assets/random_vis_l32.png" alt="teaser" width=90% height=90%>
</p>


## Citing
If you use our work in your research, please use the following BibTeX entry.

```BibTeX
@article{yu2024an,
  author    = {Qihang Yu and Mark Weber and Xueqing Deng and Xiaohui Shen and Daniel Cremers and Liang-Chieh Chen},
  title     = {An Image is Worth 32 Tokens for Reconstruction and Generation},
  journal   = {arxiv: 2406.07550},
  year      = {2024}
}
```

## Acknowledgement

[MaskGIT](https://github.com/google-research/maskgit)

[Taming-Transformers](https://github.com/CompVis/taming-transformers)

[Open-MUSE](https://github.com/huggingface/open-muse)

[MUSE-Pytorch](https://github.com/baaivision/MUSE-Pytorch)
