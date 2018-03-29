from nnUtils import *

# WRN50-2 bottleNeck for alexnet
## for WRN50-2: K=2.N= ???
K=2
N=3
model = Sequential([
    SpatialConvolution(64,7,7,2,2,padding='SAME',bias=False),
    BatchNormalization(),
    ReLU(),
    SpatialMaxPooling(3,3,2,2),
    Block(64,3,3,K=K,N=N,padding='SAME',bias=False,fixShapeMethod='conv'),
    Block(128,3,3,2,2,K=K,N=N,padding='SAME',bias=False,fixShapeMethod='conv'),
    Block(256,3,3,2,2,K=K,N=N,padding='SAME',bias=False,fixShapeMethod='conv'),
    Block(512,3,3,2,2,K=K,N=N,padding='SAME',bias=False,fixShapeMethod='conv'),
    SpatialAveragePooling(7,7,1,1),
    Affine(1001,bias=False)
])
