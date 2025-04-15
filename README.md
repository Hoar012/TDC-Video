## Multimodal Long Video Modeling Based on Temporal Dynamic Context

### [Paper](https://arxiv.org/abs/2504.10443) | [Project Page](https://hoar012.github.io/TDC-Project/) | [Model](https://github.com/Hoar012/TDC-Video)


## Unified Multimodal Long Video Understanding
| <img src="./images/teaser.png" alt="MM-Video" width="600"> |
|:--:|

## Framework of Temporal Dynamic Context Compression
| ![TDC](./images/framework.png) |
|:--:|
| Architecture of Our Multimodal Video Encoder. We first extract features for each second of the video, including both visual and corresponding audio tokens. The first frame is selected as the static frame, and a Q-Former is used to perform Temporal Dynamic Context compression based on its relationship with subsequent frames, resulting in K compressed tokens per frame. The final video representation consists of all static frame tokens and multimodal video context. |


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