# DL Satellite Stereo Matching Algorithm
This is a non-official pytorch implementation of deep learning stereo matching network and make a slight modification to adapt it to satellite image.<br />

Now support:
- [GCNet](https://arxiv.org/abs/1703.04309)
- [PSMNet](https://arxiv.org/abs/1803.08669)
- [StereoNet](https://arxiv.org/abs/1807.08865)
- [GwcNet](https://arxiv.org/abs/1903.04025)
- PSMNet + StereoNet edge refinement module
- PSMNet + Gwc volume

### Dataset
The dataloader only support `DFC2019` and `WHU-Stereo` dataset<br />
The dataloader can support both 3channel 8bit and 1channel 16bit image. <br />
You can train the model by using `train.py`

```
python train.py --model_name PSMNet --dataset_name DFC2019 --epoch 10
```
    
You can use tensorboard to monitoring the loss and learning rate during training.

```
tensorboard --logdir YOUR_LOGDIR
```

### Environment
- cudatoolkit               11.8
- torch                     2.1.1
- torchvision               0.16.1
- numpy                     1.24.1
- matplotlib                3.2.2
- opencv-python             4.8.1.78

### Inference
You can test a single pair of stereo image using the notebook `demo.ipynb`<br />
Or use the `validation.py` to test the hold test set
