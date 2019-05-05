# Whale Identification 

## Scripts/File Introduction 
- train.py:
	- initialize model from `torchvision.model`
	- train for 30 epochs
	- save the model parameters
- test.py
	- load model
	- predict labels based on test data
	- output `submission.csv`
- whaleData.py
	- define whale data set: train and test
- img_transform.py
	- define functions of images transform
	- given a threshold, if the random meet the threshold, then do the transform.
- output_label.py
	- test script to output the `label.csv` file
- random_show_whale.py
	- just for testing ..
- oversample 
	- oversample the train data set and try to solve the problem of unbalance 

## Modules used
- cv2 to do image processing
- torch
- torchvision

## Models 
- Several pretrained models from [PyTorch](https://pytorch.org/docs/): Resnet18, Resnet50, Resnet101
- 


## Reference:
1. jiqizhixin [识别座头鲸，Kaggle竞赛第一名解决方案解读](https://zhuanlan.zhihu.com/p/58496385)
2. kaggle [Humpback Whale Identification](https://www.kaggle.com/c/humpback-whale-identification/overview)
3. 3rd solution [3rd place solution with code: ArcFace](https://www.kaggle.com/c/humpback-whale-identification/discussion/82484#latest-502552)
4. @radel [https://github.com/radekosmulski/whale]