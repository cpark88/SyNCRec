# Pacer and Runner: Cooperative Learning Framework between Single- and Cross-Domain Sequential Recommendation (SIGIR '24; Best Paper Honorable Mention)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fcpark88%2FSyNCRec&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

## Overview
***
Code for our SIGIR 2024 (<https://sigir-2024.github.io>) Paper "Pacer and Runner: Cooperative Learning Framework between Single- and Cross-Domain Sequential Recommendation." 
![model_arch](https://github.com/cpark88/SyNCRec/blob/main/syncrec_github_arch.png)
We referred to the source code of S3Rec (<https://github.com/RUCAIBox/CIKM2020-S3Rec/tree/master>).

## Abstract
***
Cross-Domain Sequential Recommendation (CDSR) improves recommendation performance by utilizing information from multiple domains, which contrasts with Single-Domain Sequential Recommendation (SDSR) that relies on a historical interaction within a specific domain. However, CDSR may underperform compared to the SDSR approach in certain domains due to negative transfer, which occurs when there is a lack of relation between domains or different levels of data sparsity. To address the issue of negative transfer, our proposed CDSR model estimates the degree of negative transfer of each domain and adaptively assigns it as a weight factor to the prediction loss, to control gradient flows through domains with significant negative transfer. To this end, our model compares the performance of a model trained on multiple domains (CDSR) with a model trained solely on the specific domain (SDSR) to evaluate the negative transfer of each domain using our asymmetric cooperative network. In addition, to facilitate the transfer of valuable cues between the SDSR and CDSR tasks, we developed an auxiliary loss that maximizes the mutual information between the representation pairs from both tasks on a per-domain basis. This cooperative learning between SDSR and CDSR tasks is similar to the collaborative dynamics between pacers and runners in a marathon. Our model outperformed numerous previous works in extensive experiments on two real-world industrial datasets across ten service domains. We also have deployed our model in the recommendation system of our personal assistant app service, resulting in 21.4\% increase in click-through rate compared to existing models, which is valuable to real-world business.


## Usage
***

```bash
bash train.sh
```


## Cite
If you use our codes for your research, cite our paper:

```
@inproceedings{park2024pacer,
  title={Pacer and Runner: Cooperative Learning Framework between Single-and Cross-Domain Sequential Recommendation},
  author={Park, Chung and Kim, Taesan and Yoon, Hyungjun and Hong, Junui and Yu, Yelim and Cho, Mincheol and Choi, Minsung and Choo, Jaegul},
  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={2071--2080},
  year={2024}
}
```


[![Top Langs](https://github-readme-stats.vercel.app/api/top-langs/?username=cpark88)](https://github.com/anuraghazra/github-readme-stats)
