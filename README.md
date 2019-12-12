# code for the paper 'Few-shot Breast Cancer Metastases Classification via Unsupervised Cell Ranking'
our method are implemented in the pytorch framework,and the data processing are mainly referred to [here](https://github.com/arjunvekariyagithub/camelyon16-grand-challenge)

### pre-processing
extract patches for training

### post-processing
generate heatmap and extract features from heatmap

### ranking.py
train the ranking model

### classification.py
train the supervised model

### finetune.py
finetune the pre-train ranking model


@article{title={Few-shot Breast Cancer Metastases Classification via Unsupervised Cell Ranking},
  author={Jiaojiao Chen, Jianbo Jiao, Shengfeng He, Guoqiang Han, and Jing Qin},
  journal={TCBB},
  year={2019}
}
