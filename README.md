# ROI-Extraction

>ROI extraction is an active but challenging task in remote sensing because of the complicated landform, the complex boundaries and the requirement of annotations. Weakly supervised learning (WSL) aims at learning a mapping from input image to pixel-wise prediction under image-wise labels, which can dramatically decrease the labor cost. However, due to the imprecision of labels, the accuracy and time consumption of WSL methods are relatively unsatisfactory. In this paper, we propose a two-step ROI extraction based on contractive learning. Firstly, we present to integrate multiscale Grad-CAM to obtain pseudo pixelwise annotations with well boundaries. Then, to reduce the compact of misjudgments in pseudo annotations, we construct a contrastive learning strategy to encourage the features inside ROI as close as possible and separate background features from foreground features. Comprehensive experiments demonstrate the superiority of our proposal.

[arxiv paper](https://arxiv.org/abs/2305.05887) [This paper has been accepted in IGARSS 2023](https://ieeexplore.ieee.org/document/10283054) 

## Train
```python
python train.py --config_path [your config path]
```

## Test
```python
python predict.py --config_path [your config path]
```
