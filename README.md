# 1D Visual Tokenization and Generation

This repo hosts the code and models for the following projects:

- RAR: [Randomized Autoregressive Visual Generation](https://yucornetto.github.io/projects/rar.html)

- TiTok: [An Image is Worth 32 Tokens for Reconstruction and Generation](https://yucornetto.github.io/projects/titok.html)


## Short Intro on [Randomized Autoregressive Visual Generation](https://arxiv.org/abs/2406.07550)

RAR is a an autoregressive (AR) image generator with full compatibility to language modeling. It introduces a randomness annealing strategy with permuted objective at no additional cost, which enhances the model's ability to learn bidirectional contexts while leaving the autoregressive framework intact. RAR sets a FID score 1.48, demonstrating state-of-the-art performance on ImageNet-256 benchmark and significantly outperforming prior AR image generators.

<p>
<img src="assets/rar_overview.png" alt="teaser" width=90% height=90%>
</p>
<p>
<img src="assets/perf_comp.png" alt="teaser" width=90% height=90%>
</p>

See more details at [README_RAR](README_RAR.md).

## Short Intro on [An Image is Worth 32 Tokens for Reconstruction and Generation](https://arxiv.org/abs/2406.07550)

We present a compact 1D tokenizer which can represent an image with as few as 32 discrete tokens. As a result, it leads to a substantial speed-up on the sampling process (e.g., **410 × faster** than DiT-XL/2) while obtaining a competitive generation quality.

<p>
<img src="assets/titok_teaser.png" alt="teaser" width=90% height=90%>
</p>
<p>
<img src="assets/speed_vs_perf.png" alt="teaser" width=90% height=90%>
</p>

See more details at [README_TiTok](README_TiTok.md).

## Updates
- 11/04/2024: We release the [tech report](https://arxiv.org/abs/2411.00776) and code for RAR models.
- 10/16/2024: We update a set of TiTok tokenizer weights trained with an updated single-stage recipe, leading to easier training and better performance. We release the weight of different model size for both VQ and VAE variants TiTok, which we hope could facilitate the research in this area. More details will be available in a tech report later. 
- 09/25/2024: TiTok is accepted by NeurIPS 2024.
- 09/11/2024: Release the training codes of generator based on TiTok. 
- 08/28/2024: Release the training codes of TiTok.
- 08/09/2024: Better support on loading pretrained weights from huggingface models, thanks for the help from [@NielsRogge](https://github.com/NielsRogge)！
- 07/03/2024: Evaluation scripts for reproducing the results reported in the paper, checkpoints of TiTok-B64 and TiTok-S128 are available.
- 06/21/2024: Demo code and TiTok-L-32 checkpoints release. 
- 06/11/2024: The [tech report](https://arxiv.org/abs/2406.07550) of TiTok is available.


## Installation
```shell
pip3 install -r requirements.txt
```

## Citing
If you use our work in your research, please use the following BibTeX entry.

```BibTeX
@article{yu2024randomized,
  author    = {Qihang Yu and Ju He and Xueqing Deng and Xiaohui Shen and Liang-Chieh Chen},
  title     = {Randomized Autoregressive Visual Generation},
  journal   = {arXiv preprint arXiv:2411.00776},
  year      = {2024}
}
```

```BibTeX
@inproceedings{yu2024an,
  author    = {Qihang Yu and Mark Weber and Xueqing Deng and Xiaohui Shen and Daniel Cremers and Liang-Chieh Chen},
  title     = {An Image is Worth 32 Tokens for Reconstruction and Generation},
  journal   = {NeurIPS},
  year      = {2024}
}
```

## Acknowledgement

[MaskGIT](https://github.com/google-research/maskgit)

[Taming-Transformers](https://github.com/CompVis/taming-transformers)

[Open-MUSE](https://github.com/huggingface/open-muse)

[MUSE-Pytorch](https://github.com/baaivision/MUSE-Pytorch)
