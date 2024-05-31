[![Replicate](https://replicate.com/schananas/grounded_sam/badge)](https://replicate.com/schananas/grounded_sam/badge)

# Grounded Sam

Implementation of [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) & [Segment Anything](https://github.com/facebookresearch/segment-anything), and it allows masking based on prompt, which is useful for programmed inpainting.

This project combines strengths of two different models in order to build a very powerful pipeline for solving complex masking problems.

Segment-Anything aims to segment everything in an image, which needs prompts (as boxes/points/text) to generate masks.

Grounding DINO, a strong zero-shot detector which, is capable of to generate high quality boxes and labels with free-form text.

On top of Segment-Anything & Grounding DINO this project adds possibility to prompt multiple masks and combine them into one, as well to subtract negative mask for fine grain control.


# Setup
If you don't have a GPU that can run these models, use a Lamda Instance (it's the only service that allows you to run cog)

1. Follow instructions on cog repo to install cog onto the instance: https://github.com/replicate/cog/
2. Clone this repo and cd into it
3. Install the models (by running the script in /scripts folder or following the instructions to manually install below)
4. run ```sudo cog predict```


To test models locally, you need to download the models / repos:
```
mkdir models
cd models
git-lfs install
git clone https://huggingface.co/google/owlv2-base-patch16-ensemble
git clone https://huggingface.co/facebook/sam-vit-base
```


## Citation

```BibTex
@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}
```