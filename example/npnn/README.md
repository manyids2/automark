# NPNN: NumPy Neural Network

```
    _            __
  / /|         /  /|          ___                           
 / / |        /  / |        /   /|                      #   
/_/  |       /__/  |       /   / |          ____        #  
| |  |       |  |  |      /___/  |        /    /|       #       #
| |  |       |  |  |      |   |  |       /____/ |       #       #
| | -------> |  | ------> |   | ------>  |    | ------> # ----> #
| |  |       |  |  |      |   |  |       |____|/        #       #
| |  |       |  |  |      |   | /                       #       #
| | /        |  | /       |___|/                        #
|_|/         |__|/                                      #
```

## MNIST ConvNet Example
First of all, import all required modules:

```python
from npnn.models import SequentialNN
from npnn.optimizers import SGD
from npnn.losses import Hinge
from npnn.functions.regularizer import l2_regularizer
from npnn.layers import *
```

Build a neural network:

```python
nn = SequentialNN()

nn.add(ConvLayer(1, 2, 3)) # 28 28 2
nn.add(ReLU())
nn.add(MaxPool2x2()) # 14 14 2

nn.add(ConvLayer(2, 4, 3)) # 14 14 4
nn.add(ReLU())
nn.add(MaxPool2x2()) # 7 7 4

nn.add(FlattenLayer())
nn.add(Dense(196, 32))
nn.add(ReLU())
nn.add(Dense(32, 1))

loss = Hinge()
optimizer = SGD(nn)
```

Train your model:

```python
for epoch in range(num_epochs):
    
    for x_batch, y_batch in iterate_minibatches(x_train, y_train, batch_size):
        # predict the target value
        y_pred = nn.forward(x_batch)
        # compute the gradient of the loss
        loss_grad = loss.backward(y_pred, y_batch)
        # perform backprop
        nn.backward(x_batch, loss_grad)
        # update the params
        optimizer.update_params()

```

