import jax
from jax import numpy as jnp
from flax import linen as fnn


class Encoder(fnn.Module):
    """
    This code defines a class called Encoder, 
    which is a component of a UNet architecture
    commonly used in image segmentation tasks. 
    The Encoder class takes as input an image 
    tensor and produces a set of skip connections, 
    which are used by the Decoder network to 
    generate a segmentation mask.

    The Encoder network consists of a series of 
    convolutional layers with increasing numbers 
    of channels and decreasing spatial dimensions.
    Each set of features is passed through batch 
    normalization and activation functions. The 
    network also includes max-pooling and dropout 
    layers in the higher-resolution feature maps 
    to reduce overfitting and improve generalization.

    The Encoder class has two class variables: 
    features and training. features is an integer 
    that specifies the number of channels in the 
    convolutional layers of the network. training 
    is a boolean that specifies whether the network
    is in training mode or not.

    The __call__ method of the Encoder class is defined
    using the @fnn.compact decorator, which is used in
    the JAX library to define neural networks in a 
    concise manner. The __call__ method takes the input
    image tensor x as input and returns a list of skip 
    connections as output.

    The Encoder network applies a series of convolutional 
    layers to the input image, producing a set of feature 
    maps with different spatial dimensions and channel sizes. 
    The network then applies batch normalization and ReLU 
    activation functions to each set of features. Max-pooling
    is used in the lower-resolution feature maps to reduce 
    the spatial dimensions and provide translation invariance.
    Dropout is used in the higher-resolution feature maps to 
    reduce overfitting and improve generalization.

    The output of the Encoder network is a list of feature maps 
    at multiple scales. The feature maps in the list are in 
    decreasing order of spatial dimensions and increasing order 
    of channel sizes. The list of feature maps is used as input 
    to the Decoder network, which upsamples and combines the 
    features to produce a segmentation mask.

    Overall, the Encoder class defines the feature extraction 
    and downsampling operations of the UNet architecture, 
    which are crucial for producing accurate segmentation 
    masks in image segmentation tasks.

    """
    features: int = 64
    training: bool = True

    @fnn.compact
    def __call__(self, x):
        skips = []
        for i in range(5):
            z = fnn.Conv(self.features * 2 ** i, kernel_size=(3, 3))(x)
            z = fnn.relu(z)
            z = fnn.Conv(self.features * 2 ** i, kernel_size=(3, 3))(z)
            z = fnn.BatchNorm(use_running_average=not self.training)(z)
            z = fnn.relu(z)
            if i < 4:
                x = fnn.max_pool(z, window_shape=(2, 2), strides=(2, 2))
            if i == 3 or i == 4:
                z = fnn.Dropout(0.5, deterministic=False)(z)
            skips.append(z)
            
        return skips


class Decoder(fnn.Module):
    """ This code defines a class called Decoder, 
        which is a component of a UNet architecture 
        commonly used in image segmentation tasks. 
        The Decoder class takes as input a set of 
        skip connections z_skips, which are produced 
        by the Encoder network and contain feature 
        maps at multiple scales.
        
        The Decoder class has two class variables: 
        features and training. features is an integer 
        that specifies the number of channels in the 
        convolutional layers of the network. 
        training is a boolean that specifies whether 
        the network is in training mode or not.
        
        The __call__ method of the Decoder class is
        defined using the @fnn.compact decorator, which 
        is used in the JAX library to define neural networks
        in a concise manner. The __call__ method takes 
        the set of skip connections z_skips as input and 
        returns a segmentation mask as output.

        The Decoder network first selects the last set of
        features from z_skips, and then iteratively upsamples
        and concatenates the feature maps from the other sets 
        of skip connections with the current set of features. 
        The concatenated features are then passed through a 
        series of convolutional layers and activation functions 
        to produce the final segmentation mask.

        The output of the Decoder network is a single-channel 
        tensor with the same spatial dimensions as the input 
        images. Each pixel in the output tensor represents the
        probability that the corresponding pixel in the input 
        image belongs to the object or class being segmented. 
        The sigmoid activation function is used to ensure that 
        the output values are in the range [0, 1].

        Overall, the Decoder class defines the upsampling and 
        feature combination operations of the UNet architecture, 
        which are crucial for producing accurate segmentation masks
        in image segmentation tasks.

    """
    features: int = 64
    training: bool = True

    @fnn.compact
    def __call__(self, z_skips):
        z = z_skips[-1]
        for i in range(4):
            z = jax.image.resize(z, 
                                 shape=(z.shape[0], z.shape[1] * 2, z.shape[2] * 2, z.shape[3]), 
                                 method='nearest')
            z = fnn.Conv(self.features * 8, kernel_size=(2, 2))(z)
            z = fnn.relu(z)
            z = jnp.concatenate([z_skips[-i-2], z], axis=3)
            z = fnn.Conv(self.features * 8, kernel_size=(3, 3))(z)
            z = fnn.relu(z)
            z = fnn.Conv(self.features * 8, kernel_size=(3, 3))(z)
            z = fnn.BatchNorm(use_running_average=not self.training)(z)
            z = fnn.relu(z)

        z = fnn.Conv(1, kernel_size=(1, 1))(z)
        z = fnn.sigmoid(z)

        return z


class UNet(fnn.Module):
    """This code defines a class called UNet, 
       which is a type of convolutional neural 
       network commonly used for image segmentation 
       tasks. The UNet architecture consists of 
       an encoder and a decoder network, which are 
       connected through skip connections.
       The class has two class variables: features 
       and training. features is an integer that 
       specifies the number of channels in the 
       convolutional layers of the network. training 
       is a boolean that specifies whether the 
       network is in training mode or not.

    Args:
        fnn (__call__): 
        The __call__ method of the UNet class is 
        defined using the @fnn.compact decorator, 
        which is used in the JAX library to define 
        neural networks in a concise manner.
        The __call__ method takes an input tensor x, 
        which is passed through the Encoder network 
        to produce a set of skip connections. 
        These skip connections are then passed through 
        the Decoder network to produce the final
        output tensor y.
        The Encoder network is responsible for 
        downsampling the input tensor and extracting 
        high-level features, while the Decoder network 
        is responsible for upsampling the features 
        and producing the final segmentation mask.

    Returns:
        _type_: 
        The return value of the __call__ method in 
        this code is a tensor y, which represents 
        the output of the UNet architecture after 
        processing the input tensor x. The exact
        shape and data type of y will depend on the 
        specific implementation of the Decoder 
        network and the nature of the image segmentation
        task that the UNet is being used for.
        In general, the output tensor y will likely 
        be a multi-channel tensor with the same spatial 
        dimensions as the input tensor x. Each channel 
        of y may represent a different segmentation class 
        or feature of the input image, depending on the 
        task at hand. The data type of y will likely 
        be a floating-point type, such as float32, 
        to allow for the computation of gradients 
        during backpropagation.
    
    Overall, this code defines a UNet architecture 
    that can be used for image segmentation tasks. 
    By changing the value of the features and 
    training variables, the architecture can be 
    customized to fit different input data and training regimes.
    """
    features: int = 64
    training: bool = True

    @fnn.compact
    def __call__(self, x):
        z_skips = Encoder(self.training)(x)
        y = Decoder(self.training)(z_skips)

        return y