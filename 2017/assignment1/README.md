## TODO

### Part 1
- Implement 1-Stream Variant of AlexNet (tfutils.model.alexnet ?) [DONE]
- Implement one or more smaller variants of deep neural network, with fewer layers and fewer filters per layer (LeNet adapted to ImageNet) [DONE]
- Implement Inception v3 or VGG
- implement a hard-coded 1-layer CNN where the filter kernels are a fixed (untrained) Gabor filterbank, implemented as a Tensorflow model.[DONE]
- Edit `train_imagenet.py` (basically configuration). [DONE]

### Part 2
- 
-
-
-

### AlexNet Reqs:

| Architecture Requirement | Satisfied?|
|-----|-----|
|the net contains eight layers with weights; the first five are convolutional and the remaining three are fully- connected | Yes |
| output of the last fully-connected layer is fed to a 1000-way softmax which produces a distribution over the 1000 class labels. | No |
| The kernels of the second, fourth, and fifth convolutional layers are connected only to those kernel maps in the previous layer which reside on the same GPU (see Figure 2). | Not sure we need this... |
| The kernels of the third convolutional layer are connected to all kernel maps in the second layer. | Not sure if this applies ...| 
|The neurons in the fully- connected layers are connected to all neurons in the previous layer. | Yes | 
|Response-normalization layers follow the first and second convolutional layers | No (But apparently this is optional) |
| Max-pooling layers, of the kind described in Section 3.4, follow both response-normalization layers as well as the fifth convolutional layer | Yes | 
| The first convolutional layer filters the 224×224×3 input image with 96 kernels of size 11×11×3 with a stride of 4 pixels | Yes | 
| The second convolutional layer takes as input the (response-normalized and pooled) output of the first convolutional layer and filters it with 256 kernels of size 5×5×48. | Yes (we use a stride of 1, the paper says nothing on how many strides).| 
| The third, fourth, and fifth convolutional layers are connected to one another without any intervening pooling or normalization layers. | Yes |
| The third convolutional layer has 384 kernels of size 3 × 3 × 256 connected to the (normalized, pooled) outputs of the second convolutional layer | Yes (no mention of strides)|
| The fourth convolutional layer has 384 kernels of size 3 × 3 × 192 | Yes |
| the fifth convolutional layer has 256 kernels of size 3×3×192 | Yes |
| The fully-connected layers have 4096 neurons each. | Yes | 

| Training Requirement | Satisfied?|
|-----|-----|
| We employ two distinct forms of data augmentation, both of which allow transformed images to be produced from the original images with very little computation | Unlikely...|
| The recently-introduced technique, called “dropout” [10], consists of setting to zero the output of each hidden neuron with probability 0.5. | Yes |  
| At test time, we use all the neurons but multiply their outputs by 0.5 | No (not sure we need to) | 
| stochastic gradient descent with a batch size of 128 examples | No (we use 256) |
| momentum of 0.9 | Yes | 
| weight decay of 0.0005 | No (the excercise asks for a piecewise linear) |
| heuristic which we followed was to divide the learning rate by 10 when the validation error rate stopped improving with the current learning rate. The learning rate was initialized at 0.01 and reduced three times prior to termination. | No (the excercise asks for a piecewise linear) | 
| We trained the network for roughly 90 cycles through the training set of 1.2 million images | Yes | 

### V1 Reqs:

| Architecture Requirement | Satisfied?|
|-----|-----|
| Image preparation. First we converted the input image to grayscale |  Yes |
| resized by bicubic interpolation the largest edge to a fixed size (150 pixels for Caltech datasets) while preserving its aspect ratio. | N/A |
| The mean was subtracted from the resulting two-dimensional image and we divided it by its standard deviation | No |
| For each pixel in the input image, we
subtracted themean of the pixel values in a fixed window (333pixels, centered on the pixel), and we divided this value by the euclidean norm of the resulting 9-dimensional vector (333 window) if the norm was greater than 1 | Used the already implemented LRN instead |
| Linear filtering with a set of Gabor filters. We convolved the normalized
images with a set of two-dimensional Gabor filters of fixed size (433 43 pixels), spanning 16 orientations (equally spaced around the clock) and six spatial frequencies (1/2, 1/3, 1/4, 1/6, 1/11, 1/18 cycles/pixel) with a fixed Gaussian envelope (standard deviation of 9 cycles/pixel in
both directions) and fixed phase (0) for a total of N¼96 filters | Yes |
|Each filter had zero-mean and euclidean norm of one  | No | 
| Thresholding and saturation. The output of each Gabor filter was
passed through a standard output non-linearity—a threshold and response saturation.|  ReLu instead| 
| The result of the Gabor filtering
was a three-dimensional matrix of size H3W3 N where each two- dimensional slice (H3W) is the output of each Gabor filter type. For each filter output, we subtracted the mean of filter outputs in a fixed spatial window (3 3 3 pixels, centered) across all orientations and spatial scales (total of 864 elements).| Implemented LRN instead | 
| the dimensionality- reduced training data were used to train a linear support vector machine (SVM) using libsvm-2.82 [32] | Used three fully connected layers instead. |