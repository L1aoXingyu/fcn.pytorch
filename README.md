# fcn.pytorch

PyTorch implementation of [Fully Convolutional Networks](https://github.com/shelhamer/fcn.berkeleyvision.org), main code modified from [pytorch-fcn](https://github.com/wkentaro/pytorch-fcn).

### Requirements
- pytorch
- torchvision
- [ignite](https://github.com/pytorch/ignite)
- [yacs](https://github.com/rbgirshick/yacs)
- [tensorboardX](https://github.com/lanpa/tensorboardX)
- tensorflow (for tensorboard)

### Get Started
The designed architecture follows this guide [PyTorch-Project-Template](https://github.com/L1aoXingyu/PyTorch-Project-Template), you can check each folder's purpose by yourself.

### Training
Most of the configuration files that we provide are in folder `configs`. You just need to modify `dataset root`, `vgg model weight` and `output directory`. There are a few possibilities:

#### 1. Modify configuration file and run
You can modify `train_fcn32s.yml` first and run following code

```bash
python3 tools/train_net.py --config_file='configs/train_fcn32s.yml'
```

#### 2. Modify the cfg parameters
You can change configuration parameter such as learning rate or max epochs in command line.

```bash
python3 tools/train_net.py --config_file='configs/train_fcn32s.yml' SOLVER.BASE_LR 0.0025 SOLVER.MAX_EPOCHS 8
``` 
 
### Results



