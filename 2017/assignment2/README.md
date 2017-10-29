# Assignment 2

## Network Training

Train on CIFAR-10: 
- A shallow bottleneck convolutional autoencoder model (John)
- A pooled version of a shallow bottleneck convolutional autoencoder model (John)
- Sparse  version of  a shallow bottleneck convolutional autoencoder model (John)
- Sparse version of a pooled version of a shallow bottleneck convolutional autoencoder model (John)
- A deep symmetric convolutional autoencoder (Ramon)
- A Variational Autoencoder (Ramon)
- The Colorful Image Colorization network (Ramon)

Train on Imagenet:
- The Colorful Image Colorization network (TBA)
- Best of the other networks (TBA)


## Network Evaluation

Evaluations to run (Sophia):
- Perform the task generalization
- RSA 
- neural fitting
- filter visualization evaluations
- linear classifier from the features at that layer to solve Imagenet

Do this for each trained model, for every relevant time step, for each relevant layer.


## Network descriptions

### Colorful Image Colorization (Zhang et al. 2016)

- *X* spatial resolution of output 
- *C* number of channels  of  output
- *S* computation  stride,  values  greater  than  1  indicate  downsampling following  convolution,  values less than 1 indicate upsampling preceding convolution
- *D* kernel dilation
- *Sa* accumulated stride across all preceding layers (product over all strides in previous layers);
- *De* effective dilation of the layer with respect to the input (layer dilation times accumulated stride);
- *BN* whether BatchNorm layer was used after layer
- *L* whether a 1x1 conv and cross-entropy loss layer was imposed

| Layer | X | C | S | D | Sa | De | BN | L |
|:------|:--|:--|:--|:--|:---|:---|:---|:--|
| Data  |224|  3| - | - | -  |  - |  - | - |
|conv1_1|224| 64| 1 | 1 | 1  |  1 |  - | - |
|conv1_2|112| 64| 2 | 1 | 1  |  1 |  Y | - |
|conv2_1|112|128| 1 | 1 | 2  |  2 |  - | - |
|conv2_2| 56|128| 2 | 1 | 2  |  2 |  Y | - |
|conv3_1| 56|256| 1 | 1 | 4  |  4 |  - | - |
|conv3_2| 56|256| 1 | 1 | 4  |  4 |  - | - |
|conv3_3| 28|256| 2 | 1 | 4  |  4 |  Y | - |
|conv4_1| 28|512| 1 | 1 | 8  |  8 |  - | - |
|conv4_2| 28|512| 1 | 1 | 8  |  8 |  - | - |
|conv4_3| 28|512| 1 | 1 | 8  |  8 |  Y | - |
|conv5_1| 28|512| 1 | 2 | 8  |  16|  - | - |
|conv5_2| 28|512| 1 | 2 | 8  |  16|  - | - |
|conv5_3| 28|512| 1 | 2 | 8  |  16|  Y | - |
|conv6_1| 28|512| 1 | 2 | 8  |  16|  - | - |
|conv6_2| 28|512| 1 | 2 | 8  |  16|  - | - |
|conv6_3| 28|512| 1 | 2 | 8  |  16|  Y | - |
|conv7_1| 28|256| 1 | 1 | 8  |   8|  - | - |
|conv7_2| 28|256| 1 | 1 | 8  |   8|  - | - |
|conv7_3| 28|256| 1 | 1 | 8  |   8|  Y | - |
|conv8_1| 56|128| .5| 1 | 4  |   4|  - | - |
|conv8_2| 56|128| 1 | 1 | 4  |   4|  - | - |
|conv8_3| 56|128| 1 | 1 | 4  |   4|  - | Y |