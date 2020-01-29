TSN
-

Introduction
-
TSN is the abstraction of Temporal Segment Network. It can do action recognition. Given a video containing an action, it can recognize the action that is happening. Action recognition is basically video classification. The way TSN does action recognition is very similiar to image classification. First, it picks several segments of the video, which are actually a set of images. Then it does 'image classification' on each of the segment, and averages the results to get the result of the video. Then it can determine the label of the video.

The paper where TSN comes from is [Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](http://arxiv.org/abs/1608.00859). The codes in this repository are basically based on [yjxiong
/tsn-pytorch](https://github.com/yjxiong/tsn-pytorch), which is also the origin code of the paper. [yjxiong/temporal-segment-networks](https://github.com/yjxiong/temporal-segment-networks) is the caffe version code.

Dependency
-
- Ubuntu 
- PyTorch 1.4.0
- Pillow 7.0.0
- a GPU

How To Use
-
1. The model in this repository is trained on ucf101, so you need to download ucf101 first, you can download it from [here](https://www.crcv.ucf.edu/data/UCF101.php). Then you need to prepare the dataset following the instructions in [yjxiong/temporal-segment-networks](https://github.com/yjxiong/temporal-segment-networks). The two most important things are frames and video lists.

2. Download this repository.

        git clone --recursive https://github.com/Ruiyang-061X/TSN.git

3. Using the frames and video lists, you can train the model. Excute the following command to train the model. The trained models are saved in `trained_model/`

        python3 train.py --dataset ucf101 --modality RGB --trainset YOUR_TRAIN_VIDEO_LIST --validationset YOUR_VALIDATION_VIDEO_LIST --base_model BNInception --n_segment 3 --consensus_type avg --dropout 0.8 --epoch 80 --batch_size 4 --lr 0.001 --lr_step 30 60 --clip_gradient 20

    You can change --modality RGB to --modality RGBDiff or --modality Flow to train on the RGBDiff or optical flow version of the dataset.

Result
-

The following results are results on ucf101.

name | base_model | modality | accuracy@1 | accuracy@5
-|-|-|-|-
BNInception_RGB | BNInception | RGB | 69.05 | 84.40 |
---

The trained model can be downloaded from [BaiduNetdisk](https://pan.baidu.com/s/16Vl011D-yiP_Kw8jRzJbBA), the code is 1rr9.