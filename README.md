# PAINT: Photo-realistic Fashion Design Synthesis

### [Arxiv Paper](----后期补---) | [BibTex](https://github.com/gxl-groups/PAINT#citation)

<p align='center'>  
  <img src='https://github.com/gxl-groups/PAINT/TSN/blob/pytorch/docs/PAINT.PNG' width='700'/>
</p>
<p align='center'> 
</p>
## Introduction 
In this paper, we investigate a new problem of generating a variety of multi-view fashion designs conditioned on a human pose and texture examples of arbitrary sizes, which can replace the repetitive and low-level design work for fashion designers. To solve this challenging multi-modal image translation problem, we propose a novel Photo-reAlistic fashIon desigN synThesis (PAINT) framework, which decomposes the framework into three manageable stages. 

Our proposals combine these three stage by,
1) **Layout Generative Network (LGN).** we employ a Layout Generative Network (LGN) to
transform an input human pose into a series of person semantic layouts. 
2) **Texture Synthesis Network (TSN).** we propose a Texture Synthesis Network (TSN) to synthesize textures on all transformed semantic layouts. Specifically, we design a novel attentive texture transfer mechanism
for precisely expanding texture patches to the irregular clothing regions of the target fashion designs. 
3) **Appearance Flow Network (AFN).** we leverage an Appearance Flow Network (AFN) to generate the fashion design images of other viewpoints from a single-view observation by learning 2D multi-scale appearance flow fields. 

## Requirement
* pytorch(1.1.0)
* torchvision
* numpy
* scipy
* scikit-image
* pillow
* pandas
* tqdm
* dominate
## Getting Started
### Data Preperation
We provide our **dataset files** , **extracted keypoints files** and **extracted parsing files**  for convience.

#### Fashion-Gen
- Download the Fashion-Gen dataset from [here](https://pan.baidu.com/s/1Oj3XAywMHocDsi4ASPhMtw?pwd=kad8). 
- Download train/test splits from [here](https://pan.baidu.com/s/1xrVnEYMyOAVr-rr8mbgN5w?pwd=udb2), including **all\_train.txt**, **all_test.txt**. 
- We use [OpenPose](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Cascaded_Pyramid_Network_CVPR_2018_paper.pdf) to generate keypoints.Download the keypoints files from [here](https://pan.baidu.com/s/1_WrJXbO-jUTpuneJtQ8_Fg?pwd=gutg).
- We use [Human Parser](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ke_Gong_Instance-level_Human_Parsing_ECCV_2018_paper.pdf) to generate human parsing.Download the human parsing files from [here](https://pan.baidu.com/s/1cMzODPWVlPjiyk5qMGlumA?pwd=599r).

### Train the model
* Run LGN `python train.py --dataroot dataset_root --name LGN --model stage1_gan --direction AtoB `. 
* Run TSN `python train.py `. 
* Run AFN `python train.py `.


### Test the model
* Run LGN `python test.py --dataroot dataset_root --name LGN --model stage1_gan --direction AtoB `. 
* Run TSN `python test.py`. 
* Run AFN `python test.py`. 

## Pretrained models
Download the models below and put it under `release_model/`

[LGN](https://pan.baidu.com/s/1_tlA802AEpTeGjtgQAiOjw?pwd=26us) | [TSN](https://pan.baidu.com/s/1cXQlN3MJQQoDtqdMXtbYFg?pwd=5bw2) | [AFN](https://pan.baidu.com/s/1_vZnJ1_aX037IqAIFkv6Zw?pwd=cqf5)

## TensorBoard
Visualization on TensorBoard for training is supported. 

Run TSN `tensorboard --logdir release_model --port 6006` to view training progress. 

Run AFN `tensorboard --logdir release_model --port 6006` to view training progress. 

## Example Results 

<p align='center'>  
  <img src='https://github.com/gxl-groups/PAINT/TSN/blob/pytorch/docs/result.PNG' width='700'/>
</p>
<p align='center'> 
</p>

## Citation
If any part of our paper and code is helpful to your work, please generously cite with:
```

```

