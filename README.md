# LASTED

An implementation code for paper "Generalizable Synthetic Image Detection via Language-guided Contrastive Learning"

## Table of Contents

- [Background](#background)
- [Dependency](#dependency)
- [Usage](#usage)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Background
The heightened realism of AI-generated images can be attributed to the rapid development of synthetic models, including generative adversarial networks (GANs) and diffusion models (DMs). The malevolent use of synthetic images, such as the dissemination of fake news or the creation of fake profiles, however, raises significant concerns regarding the authenticity of images. Though many forensic algorithms have been developed for detecting synthetic images, their performance, especially the generalization capability, is still far from being adequate to cope with the increasing number of synthetic models. 

<p align='center'>  
  <img src='https://github.com/HighwayWu/LASTED/blob/main/imgs/practical.jpg' width='850'/>
</p>
<p align='center'>  
  <em>The heightened realism of AI-generated images raises significant concerns regarding the image authenticity.</em>
</p>

In this work, we propose a simple yet very effective synthetic image detection method via a language-guided contrastive learning and a new formulation of the detection problem. We first augment the training images with carefully-designed textual labels, enabling us to use a joint text-image contrastive learning for the forensic feature extraction. In addition, we formulate the synthetic image detection as an identification problem, which is vastly different from the traditional classification-based approaches. It is shown that our proposed LanguAge-guided SynThEsis Detection (LASTED) model achieves much improved generalizability to unseen image generation models and delivers promising performance that far exceeds state-of-the-art competitors by +22.66% accuracy and +15.24% AUC.

<p align='center'>  
  <img src='https://github.com/HighwayWu/LASTED/blob/main/imgs/LASTED_demo.jpg' width='850'/>
</p>
<p align='center'>  
  <em>Illustration of our proposed LASTED. The training images are first augmented with the carefully-designed textual labels, and then image/text encoders are jointly trained.</em>
</p>

## Dependency
- torch 1.9.0
- clip 1.0

## Usage

1. Prepare the training/testing list file (e.g., ```annotation/Test.txt```) through ```preprocess.py```.

2. For training LASTED: set ```isTrain=1``` then ```sh main.sh```.

3. For testing LASTED: set ```isTrain=0``` and ```test_file='Test.txt'```, then ```sh main.sh```.
LASTED will detect the images listed in ```annotation/Test.txt``` and report the detection results.

**Note: The pretrained LASTED and related testing datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1xUMoS2npQs0tCno_VZPpUDDPztnD6UgD?usp=share_link). The training dataset can be downloaded from [Baidu Pan](https://pan.baidu.com/s/1ZgSiNX_Dd7cZcwiHv1ujVg?pwd=h4p7) (~120GB with 4 subsets) or [Google Drive](https://drive.google.com/drive/folders/1lPuJjUpi5QwhkBlUaphRxC2HrBNnU8Pc?usp=sharing) (~30GB with 3 subsets). To acquire the unzip code of the dataset, please fill the [License Form](https://docs.google.com/forms/d/1CZAIZEEugoGiTw8auyiU2LM8qjW0LelAWq2fuuzYAq8/) and/or contact our email (yc07912@um.edu.mo).**

## Citation

If you use this code/dataset for your research, please citing the reference:
```
@article{lasted,
  title={Generalizable Synthetic Image Detection via Language-guided Contrastive Learning},
  author={H. Wu and J. Zhou and S. Zhang},
  journal={arXiv preprint:2305.13800},
  year={2023}
}
```

## Acknowledgments
- [CLIP](https://github.com/openai/CLIP)
- Part of the dataset are from [Artstation](https://www.artstation.com), [Behance](https://www.behance.net), [CNNDetection](https://github.com/peterwang512/CNNDetection), [Dreambooth](https://www.strmr.com/), [Danbooru](https://www.gwern.net/Danbooru2021), [DMDetection](https://github.com/grip-unina/DMimageDetection), [Lexcia](https://lexica.art/), [MidJourney](https://midjourney.com/), [NightCafe](https://creator.nightcafe.studio/), [StableAI](https://stability.ai/blog/stable-diffusion-public-release), [YiJianAI](http://portal.yjai.art).
