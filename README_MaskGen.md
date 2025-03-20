# Democratizing Text-to-Image Masked Generative Models with Compact Text-Aware One-Dimensional Tokens


<div align="center">

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://tacju.github.io/projects/maskgen.html)&nbsp;&nbsp;
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](http://arxiv.org/abs/2501.07730)&nbsp;&nbsp;

</div>

<!-- <p>
<img src="assets/maskgen_teaser.png" alt="teaser" width=90% height=90%>
</p> -->

We introduce TA-TiTok, a novel text-aware transformer-based 1D tokenizer designed to handle both discrete and continuous tokens while effectively aligning reconstructions with textual descriptions.
Building on TA-TiTok, we present MaskGen, a versatile text-to-image masked generative model framework. Trained exclusively on open data, MaskGen demonstrates outstanding performance: with 32 continuous tokens, it achieves a FID score of 6.53 on MJHQ-30K, and with 128 discrete tokens, it attains an overall score of 0.57 on GenEval.


<p>
<img src="assets/tatitok_overview.png" alt="teaser" width=90% height=90%>
</p>
<p>
<img src="assets/maskgen_overview.png" alt="teaser" width=90% height=90%>
</p>


## 🚀 Contributions

#### We introduce TA-TiTok, an innovative text-aware transformer-based 1-dimensional tokenizer designed to handle both discrete and continuous tokens. TA-TiTok seamlessly integrates text information during the de-tokenization stage and offers scalability to efficiently handle large-scale datasets with a simple one-stage training recipe.

#### We propose MaskGen, a family of text-to-image masked generative models built upon TA-TiTok. The MaskGen VQ and MaskGen KL variants utilize compact sequences of 128 discrete tokens and 32 continuous tokens, respectively. Trained exclusively on open data, MaskGen achieves performance comparable to models trained on proprietary datasets, while offering significantly lower training cost and substantially faster inference speed.


## TA-TiTok Model Zoo
| arch | #tokens | Link | rFID | IS |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| VQ | 32 | [checkpoint](https://huggingface.co/turkeyju/tokenizer_tatitok_bl32_vq) | 3.95 | 219.6 |
| VQ | 64 | [checkpoint](https://huggingface.co/turkeyju/tokenizer_tatitok_bl64_vq) | 2.43 | 218.8 |
| VQ | 128 | [checkpoint](https://huggingface.co/turkeyju/tokenizer_tatitok_bl128_vq) | 1.53 | 222.8 |
| KL | 32 | [checkpoint](https://huggingface.co/turkeyju/tokenizer_tatitok_bl32_vae) | 1.53 | 222.0 |
| KL | 64 | [checkpoint](https://huggingface.co/turkeyju/tokenizer_tatitok_bl64_vae) | 1.47 | 220.7 |
| KL | 128 | [checkpoint](https://huggingface.co/turkeyju/tokenizer_tatitok_sl128_vae) | 0.90 | 227.7 |

Please note that these models are only for research purposes.

## MaskGen Model Zoo
| Model | arch | Link | MJHQ-30K FID | GenEval Overall |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| MaskGen-L | VQ | [checkpoint](https://huggingface.co/turkeyju/generator_maskgen_vq_l) | 7.74 | 0.53 |
| MaskGen-XL | VQ | [checkpoint](https://huggingface.co/turkeyju/generator_maskgen_vq_xl) | 7.51 | 0.57 |
| MaskGen-L | KL | [checkpoint](https://huggingface.co/turkeyju/generator_maskgen_kl_l) | 7.24 | 0.52 |
| MaskGen-XL | KL | [checkpoint](https://huggingface.co/turkeyju/generator_maskgen_kl_xl) | 6.53 | 0.55 |

Please note that these models are only for research purposes.

## Installation
```shell
pip3 install -r requirements.txt
```

## Get Started - TA-TiTok
```python
import torch
from PIL import Image
import numpy as np
import open_clip
import demo_util
from huggingface_hub import hf_hub_download
from modeling.tatitok import TATiTok

# Choose one from ["tokenizer_tatitok_bl32_vq", "tokenizer_tatitok_bl64_vq, tokenizer_tatitok_bl128_vq", "tokenizer_tatitok_bl32_vae", "tokenizer_tatitok_bl64_vae, tokenizer_tatitok_sl128_vae"]
tatitok_tokenizer = TATiTok.from_pretrained("turkeyju/tokenizer_tatitok_bl32_vae")
tatitok_tokenizer.eval()
tatitok_tokenizer.requires_grad_(False)

# or alternatively, downloads from hf
# hf_hub_download(repo_id="fun-research/TA-TiTok", filename="tatitok_bl32_vae.bin", local_dir="./")

# load config
# config = demo_util.get_config("configs/infer/TA-TiTok/tatitok_bl32_vae.yaml")
# tatitok_tokenizer = demo_util.get_tatitok_tokenizer(config)

clip_encoder, _, _ = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai')
del clip_encoder.visual
clip_tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
clip_encoder.transformer.batch_first = False
clip_encoder.eval()
clip_encoder.requires_grad_(False)

device = "cuda"
tatitok_tokenizer = tatitok_tokenizer.to(device)
clip_encoder = clip_encoder.to(device)

# reconstruct an image. I.e., image -> 32 tokens -> image
img_path = "assets/ILSVRC2012_val_00010240.png"
image = torch.from_numpy(np.array(Image.open(img_path)).astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
# tokenization
if tatitok_tokenizer.quantize_mode == "vq":
    encoded_tokens = tatitok_tokenizer.encode(image.to(device))[1]["min_encoding_indices"]
elif tatitok_tokenizer.quantize_mode == "vae":
    posteriors = tatitok_tokenizer.encode(image.to(device))[1]
    encoded_tokens = posteriors.sample()
else:
    raise NotImplementedError

text = ["A photo of a jay."]
text_guidance = clip_tokenizer(text).to(device)
cast_dtype = clip_encoder.transformer.get_cast_dtype()
text_guidance = clip_encoder.token_embedding(text_guidance).to(cast_dtype)  # [batch_size, n_ctx, d_model]
text_guidance = text_guidance + clip_encoder.positional_embedding.to(cast_dtype)
text_guidance = text_guidance.permute(1, 0, 2)  # NLD -> LND
text_guidance = clip_encoder.transformer(text_guidance, attn_mask=clip_encoder.attn_mask)
text_guidance = text_guidance.permute(1, 0, 2)  # LND -> NLD
text_guidance = clip_encoder.ln_final(text_guidance)  # [batch_size, n_ctx, transformer.width]

print(f"image {img_path} is encoded into tokens {encoded_tokens}, with shape {encoded_tokens.shape}")

# de-tokenization
reconstructed_image = tatitok_tokenizer.decode_tokens(encoded_tokens, text_guidance)
reconstructed_image = torch.clamp(reconstructed_image, 0.0, 1.0)
reconstructed_image = (reconstructed_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
reconstructed_image = Image.fromarray(reconstructed_image).save("assets/ILSVRC2012_val_00010240_recon.png")
```

## Get Started - MaskGen
```python
import torch
from PIL import Image
import numpy as np
import open_clip
import demo_util
from huggingface_hub import hf_hub_download
from modeling.tatitok import TATiTok
from modeling.maskgen import MaskGen_VQ, MaskGen_KL

torch.manual_seed(42)                 
torch.cuda.manual_seed(42)            
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False   

# VQ Tokenizer: load tokenizer tatitok_bl128_vq
tatitok_vq_tokenizer = TATiTok.from_pretrained("turkeyju/tokenizer_tatitok_bl128_vq")
tatitok_vq_tokenizer.eval()
tatitok_vq_tokenizer.requires_grad_(False)

# KL Tokenizer: load tokenizer tatitok_bl32_vae
tatitok_kl_tokenizer = TATiTok.from_pretrained("turkeyju/tokenizer_tatitok_bl32_vae")
tatitok_kl_tokenizer.eval()
tatitok_kl_tokenizer.requires_grad_(False)

# or alternatively, downloads from hf
# hf_hub_download(repo_id="fun-research/TA-TiTok", filename="tatitok_bl32_vae.bin", local_dir="./")

# load config
# config = demo_util.get_config("configs/infer/TA-TiTok/tatitok_bl32_vae.yaml")
# tatitok_tokenizer = demo_util.get_tatitok_tokenizer(config)

# VQ Generator: choose one from ["maskgen_vq_l", "maskgen_vq_xl"]
maskgen_vq_generator = MaskGen_VQ.from_pretrained("turkeyju/generator_maskgen_vq_xl")
maskgen_vq_generator.eval()
maskgen_vq_generator.requires_grad_(False)

# or alternatively, downloads from hf
# hf_hub_download(repo_id="fun-research/TA-TiTok", filename="maskgen_vq_xl.bin", local_dir="./")

# load config
# config = demo_util.get_config("configs/infer/MaskGen/maskgen_vq_xl.yaml")
# maskgen_vq_generator = demo_util.get_maskgen_vq_generator(config)

# KL Generator: choose one from ["maskgen_kl_l", "maskgen_kl_xl"]
maskgen_kl_generator = MaskGen_KL.from_pretrained("turkeyju/generator_maskgen_kl_xl")
maskgen_kl_generator.eval()
maskgen_kl_generator.requires_grad_(False)

# or alternatively, downloads from hf
# hf_hub_download(repo_id="fun-research/TA-TiTok", filename="maskgen_kl_xl.bin", local_dir="./")

# load config
# config = demo_util.get_config("configs/infer/MaskGen/maskgen_kl_xl.yaml")
# maskgen_kl_generator = demo_util.get_maskgen_kl_generator(config)

clip_encoder, _, _ = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai')
del clip_encoder.visual
clip_tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
clip_encoder.transformer.batch_first = False
clip_encoder.eval()
clip_encoder.requires_grad_(False)

device = "cuda"
tatitok_vq_tokenizer = tatitok_vq_tokenizer.to(device)
tatitok_kl_tokenizer = tatitok_kl_tokenizer.to(device)
maskgen_vq_generator = maskgen_vq_generator.to(device)
maskgen_kl_generator = maskgen_kl_generator.to(device)
clip_encoder = clip_encoder.to(device)

# generate an image
text = ["A cozy cabin in the middle of a snowy forest, surrounded by tall trees with lights glowing through the windows, a northern lights display visible in the sky."]
text_guidance = clip_tokenizer(text).to(device)
cast_dtype = clip_encoder.transformer.get_cast_dtype()
text_guidance = clip_encoder.token_embedding(text_guidance).to(cast_dtype)  # [batch_size, n_ctx, d_model]
text_guidance = text_guidance + clip_encoder.positional_embedding.to(cast_dtype)
text_guidance = text_guidance.permute(1, 0, 2)  # NLD -> LND
text_guidance = clip_encoder.transformer(text_guidance, attn_mask=clip_encoder.attn_mask)
text_guidance = text_guidance.permute(1, 0, 2)  # LND -> NLD
text_guidance = clip_encoder.ln_final(text_guidance)  # [batch_size, n_ctx, transformer.width]

vq_generated_tokens = maskgen_vq_generator.generate(captions=text, guidance_scale=12.0, randomize_temperature=2.0, sample_aesthetic_score=6.5, clip_tokenizer=clip_tokenizer, clip_encoder=clip_encoder)
kl_generated_tokens = maskgen_kl_generator.sample_tokens(1, clip_tokenizer, clip_encoder, num_iter=32, cfg=3.0, aes_scores=6.5, captions=text)

# de-tokenization
vq_generated_image = tatitok_vq_tokenizer.decode_tokens(vq_generated_tokens, text_guidance)
kl_generated_image = tatitok_kl_tokenizer.decode_tokens(kl_generated_tokens, text_guidance)
vq_generated_image = torch.clamp(vq_generated_image, 0.0, 1.0)
kl_generated_image = torch.clamp(kl_generated_image, 0.0, 1.0)
vq_generated_image = (vq_generated_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
kl_generated_image = (kl_generated_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
vq_generated_image = Image.fromarray(vq_generated_image).save("assets/maskgen_vq_generator_generated.png")
kl_generated_image = Image.fromarray(kl_generated_image).save("assets/maskgen_kl_generator_generated.png")
```

## Training Preparation
We use [webdataset](https://github.com/webdataset/webdataset) format for data loading. To begin with, it is needed to convert the dataset into webdataset format.

## Training
We provide example commands to train TA-TiTok as follows:
```bash
# Training for TiTok-BL32-VQ
WANDB_MODE=offline accelerate launch --num_machines=4 --num_processes=8 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network scripts/train_tatitok.py config=configs/training/TA-TiTok/tatitok_bl32_vq.yaml \
    experiment.project="tatitok_bl32_vq" \
    experiment.name="tatitok_bl32_vq_run1" \
    experiment.output_dir="tatitok_bl32_vq_run1" \

# Training for TiTok-BL32-VAE
WANDB_MODE=offline accelerate launch --num_machines=4 --num_processes=8 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network scripts/train_tatitok.py config=configs/training/TA-TiTok/tatitok_bl32_vae.yaml \
    experiment.project="tatitok_bl32_vae" \
    experiment.name="tatitok_bl32_vae_run1" \
    experiment.output_dir="tatitok_bl32_vae_run1" \

# Training for MaskGen-{VQ/KL}-{L/XL} Stage1
WANDB_MODE=offline accelerate launch --num_machines=4 --num_processes=8 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network scripts/train_maskgen.py config=configs/training/MaskGen/maskgen_{vq/kl}_{l/xl}_stage1.yaml \
    experiment.project="maskgen_{vq/kl}_{l/xl}_stage1" \
    experiment.name="maskgen_{vq/kl}_{l/xl}_stage1_run1" \
    experiment.output_dir="maskgen_{vq/kl}_{l/xl}_stage1_run1" \

# Training for MaskGen-{VQ/KL}-{L/XL} Stage2
WANDB_MODE=offline accelerate launch --num_machines=4 --num_processes=8 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network scripts/train_maskgen.py config=configs/training/MaskGen/maskgen_{vq/kl}_{l/xl}_stage2.yaml \
    experiment.project="maskgen_{vq/kl}_{l/xl}_stage2" \
    experiment.name="maskgen_{vq/kl}_{l/xl}_stage2_run1" \
    experiment.output_dir="maskgen_{vq/kl}_{l/xl}_stage2_run1" \
```
You may remove the flag "WANDB_MODE=offline" to support online wandb logging, if you have configured it.

The config can be replaced for other TA-TiTok variants.

## Visualizations
<p>
<img src="assets/maskgen_vis1.png" alt="teaser" width=90% height=90%>
</p>
<p>
<img src="assets/maskgen_vis2.png" alt="teaser" width=90% height=90%>
</p>
<p>
<img src="assets/maskgen_vis3.png" alt="teaser" width=90% height=90%>
</p>


## Citing
If you use our work in your research, please use the following BibTeX entry.

```BibTeX
@article{kim2025democratizing,
  author    = {Kim, Dongwon and He, Ju and Yu, Qihang Yu and Yang, Chenglin and Shen, Xiaohui and Kwak, Suha and Chen Liang-Chieh},
  title     = {Democratizing Text-to-Image Masked Generative Models with Compact Text-Aware One-Dimensional Tokens},
  journal   = {arXiv preprint arXiv:2501.07730},
  year      = {2025}
}
```