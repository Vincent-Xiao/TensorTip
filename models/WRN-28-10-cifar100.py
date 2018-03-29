from nnUtils import *

# WRN28-10 for cifar100
## for WRN28-10: K=10.N=(28-4)/6=4
K=10
N=4
model = Sequential([
    SpatialConvolution(16,3,3,padding='SAME'),
    BatchNormalization(),
    ReLU(),
    Block(16,3,3,K=K,N=N,padding='SAME'),
    Block(32,3,3,2,2,K=K,N=N,padding='SAME'),
    Block(64,3,3,2,2,K=K,N=N,padding='SAME'),
    SpatialAveragePooling(8,8,1,1),
    Affine(100)
])
