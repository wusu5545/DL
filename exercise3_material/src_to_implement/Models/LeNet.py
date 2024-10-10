from Optimization import *
from Layers import *
import NeuralNetwork
from servicemanager import Initialize


def build():
    # input_shape = (1, 28, 28)
    conv_stride = (1 ,1)
    num_kernel_1 = 6
    kernel_shape_1 = (1, 5, 5)
    num_kernel_2 = 16
    kernel_shape_2 = (6, 5, 5)
    pooling_stride = (2, 2)
    pooling_shape = (2, 2)
    optimizer = Optimizers.Adam(5e-4)
    optimizer.add_regularizer(Constraints.L2_Regularizer(4e-4))
    net = NeuralNetwork.NeuralNetwork(optimizer, Initializers.He(), Initializers.He())

    # Convolutional Layer 1
    conv_1 = Conv.Conv(conv_stride, kernel_shape_1, num_kernel_1)
    net.append_layer(conv_1)
    net.append_layer(ReLU.ReLU())
    # subsampling Layer 1
    pool = Pooling.Pooling(pooling_stride, pooling_shape)
    net.append_layer(pool)
    # Convolutional Layer 2
    conv_2 = Conv.Conv(conv_stride, kernel_shape_2, num_kernel_2)
    net.append_layer(conv_2)
    net.append_layer(ReLU.ReLU())
    # subsampling Layer 2
    pool = Pooling.Pooling(pooling_stride, pooling_shape)# need redefine a new pooling layer
    net.append_layer(pool)
    # Flatten Layer
    net.append_layer(Flatten.Flatten())
    # FullyConnected Layer 1
    fully_batch_size = 16 * 7 * 7
    categories_1 = 120
    fcl_1 = FullyConnected.FullyConnected(fully_batch_size,categories_1)
    net.append_layer(fcl_1)
    net.append_layer(ReLU.ReLU())
    # FullyConnected Layer 2
    categories_2 = 84
    fcl_2 = FullyConnected.FullyConnected(categories_1,categories_2)
    net.append_layer(fcl_2)
    net.append_layer(ReLU.ReLU())
    # FullyConnected Layer 3
    categories_3 = 10
    fcl_3 = FullyConnected.FullyConnected(categories_2,categories_3)
    net.append_layer(fcl_3)
    # Softmax Layer
    net.append_layer(SoftMax.SoftMax())

    net.loss_layer = Loss.CrossEntropyLoss()

    return net