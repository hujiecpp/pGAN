This is the project page associating to our work on Robust Face Sketch Synthesis:

Zhang, S., Ji, R., Hu, J., Gao, Y., Lin, C. W., "<a href=https://www.ijcai.org/proceedings/2018/0162.pdf>Robust Face Sketch Synthesis via Generative Adversarial Fusion of Priors and Parametric Sigmoid.</a>" In IJCAI, 2018.

This page contains the codes for pGAN. If you have any problem, please feel free to contact us.

# Prerequisites

* Python (2.7 or later)

* numpy

* scipy

* NVIDIA GPU + CUDA 8.0 + CuDNN v5.1

* pyTorch 0.3

# Getting Priors

* PortraitFCN [1]

* P-Net [2]

# Training & Test

After preparing the priors and training/test images:

```
./run.sh
```

The example of training phase:

```
python train.py --dataset CUHKStudent --nEpochs 20 --cuda
```

The example of test phase:

```
python test.py --train_data CUHKStudent --test_data XM2VTS --G1_model G_1_model_epoch_20.pth --G2_model G_2_model_epoch_20.pth --my_layer_model my_layer_model_epoch_20.pth --cuda
```

# Reference

[1] Shen, X., Hertzmann, A., Jia, J., Paris, S., Price, B., Shechtman, E., Sachs, I., "Automatic portrait segmentation for image stylization." In Computer Graphics Forum, 2016.

[2] Liu, S., Yang, J., Huang, C., Yang, M. H., Multi-objective convolutional learning for face labeling. In CVPR, 2015.

[3] Zhang, S., Ji, R., Hu, J., Gao, Y., Lin, C. W., "Robust Face Sketch Synthesis via Generative Adversarial Fusion of Priors and Parametric Sigmoid." In IJCAI, 2018.