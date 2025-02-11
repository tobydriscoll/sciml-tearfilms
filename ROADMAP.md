# Operator learning for mapping intensity to thickness and osmolarity

## Methods to be compared

* Fourier feature network (determine sigma values, number of modes, number and width of layers)
* POD network (number of modes, number and width of layers)
* Fourier operator network
* DeepONet

## Error metrics

* distribution of integral or relative 2-norm error
  For each test case, 
$$
\frac{\|u_{\text{true}} - u_{\text{pred}}\|_2}{\|u_{\text{true}}\|_2}
$$
* When osmolarity first reaches 2, 3, and 4 for true and predicted

## Effects to study

* Training set size
* PDE versus ODE
* predictions on the published measurements
* Characterize the most difficult test cases



