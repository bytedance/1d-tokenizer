# Democratizing Text-to-Image Masked Generative Models with Compact Text-Aware One-Dimensional Tokens


<div align="center">

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://tacju.github.io/projects/maskgen.html)&nbsp;&nbsp;
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](http://arxiv.org/abs/2501.07730)&nbsp;&nbsp;
[![demo](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/turkeyju/MaskGen)&nbsp;&nbsp;

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

## TODO

- [ ] Release training code, inference code and checkpoints of TA-TiTok (ETA: Jan 17)
- [ ] Release training code, inference code and checkpoints of MaskGen (ETA: Jan 24)


## TA-TiTok Model Zoo
| arch | #tokens | Link | rFID | IS |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| VQ | 32 | TODO | 4.09 | 215.9 |
| VQ | 64 | TODO | 2.68 | 213.5 |
| KL | 128 | TODO | 1.78 | 216.9 |
| KL | 32 | TODO | 1.53 | 222.0 |
| KL | 64 | TODO | 1.47 | 220.7 |
| KL | 128 | TODO | 0.90 | 227.7 |

Please note that these models are only for research purposes.

## MaskGen Model Zoo
| Model | arch | Link | MJHQ-30K FID | GenEval Overall |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| MaskGen-L | VQ | TODO | 7.74 | 0.53 |
| MaskGen-XL | VQ | TODO | 7.51 | 0.57 |
| MaskGen-L | KL | TODO | 7.24 | 0.52 |
| MaskGen-XL | KL | TODO | 6.53 | 0.55 |

Please note that these models are only for research purposes.

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