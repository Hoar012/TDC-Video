## Multimodal Long Video Modeling Based on Temporal Dynamic Context

### [Paper](https://arxiv.org/abs/2504.10443) | [Project Page](https://hoar012.github.io/TDC-Project/) | [Model (Coming Soon)](https://github.com/Hoar012/TDC-Video)


## News
- **2025.6.10** Release training and evaluation code.

<!-- ## Unified Multimodal Long Video Understanding
| <img src="./images/teaser.png" alt="MM-Video" width="600"> |
|:--:| -->


## 📋 Contents

- [Install](#install)
<!-- - [Models](#models)
- [Demo](#demo)
- [Data](#data)
- [Training](#Training)
- [Evaluation](#evaluation) -->

Note: 🚧 This repository is under construction 🚧 -- Please check back for updates!

## Framework of Temporal Dynamic Context Compression
| ![TDC](./images/framework.png) |
|:--:|
| Architecture of Our Multimodal Video Encoder. We first extract features for each second of the video, including both visual and corresponding audio tokens. The first frame is selected as the static frame, and a Q-Former is used to perform Temporal Dynamic Context compression based on its relationship with subsequent frames, resulting in K compressed tokens per frame. The final video representation consists of all static frame tokens and multimodal video context. |

### Install

1. Clone the repo into a local folder.

```bash
git clone https://github.com/Hoar012/TDC-Video.git
cd TDC-Video
```

2. Install packages.

```bash
conda create -n tdc python=3.10 -y
conda activate tdc
pip install -r requirements.txt
```

## BibTeX
```
@misc{hao2025multimodallongvideomodeling,
        title={Multimodal Long Video Modeling Based on Temporal Dynamic Context}, 
        author={Haoran Hao and Jiaming Han and Yiyuan Zhang and Xiangyu Yue},
        year={2025},
        eprint={2504.10443},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2504.10443}, 
  }
```


## Acknowledgement

This repository is built upon: [LLaVA](https://github.com/haotian-liu/LLaVA), [LongVU](https://github.com/Vision-CAIR/LongVU) and [StoryTeller](https://github.com/hyc2026/storyteller).