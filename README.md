# chess-SEResNET
Small (300k parameters), residual, squeeze-expand neural network trained on 35 million chess positions in TensorFlow (Python) implemented in C++.

Requires the [frugally-deep](https://github.com/Dobiasd/frugally-deep) project for the neural network backend, and includes 
the files for the header only library [surge](https://github.com/nkarve/surge) to manage the chess part.

It works pretty well, but despite using alpha-beta pruning, its not reasonably fast past depth 2. 
