# Simple DLRM

Deep Learning Recommendation Model (DLRM) for Personalization and Recommendation Systems
- Blog: https://ai.facebook.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/
- Paper: https://arxiv.org/abs/1906.00091

This repo is a simplistic implementation of DLRM model architecture. 

## Model Structure of the Sample Code
<img src="https://github.com/liyinxiao/Ranking_Papers/blob/master/assets/DLRM_batchsize128.png" width=1000 />


## Run the code
```
python3 -m venv  .env
source  .env/bin/activate
```
```
pip install torch torchvision torchaudio
```
Basic DLRM with BCE loss
```
python3 simple_dlrm.py
```
```
ls -lh dlrm_model.pth
```
DLRM with LambdaMART for NDCG optimization (ranking)
```
python3 lambdamart_dlrm.py
```
