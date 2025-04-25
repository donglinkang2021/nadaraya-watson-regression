# Nadaraya-Watson Regression

This is a simple implementation of the [Nadaraya-Watson](https://en.wikipedia.org/wiki/Kernel_regression) regression algorithm in Python. We just generate some random data of the model
$$
y = 2\sin(x) + x*{0.8} + \epsilon
$$

where x is uniformly distributed in the range [0, 5] and $\epsilon$ is a Gaussian noise with mean 0 and standard deviation 0.5.

We can use different kernels to estimate the regression function. The following kernels are implemented:

```python
def gaussian(x, sigma = 1.0):
    return torch.exp(-x**2 / (2*sigma**2))

def boxcar(x):
    return torch.abs(x) < 1.0

def constant(x):
    return 1.0 + 0 * x

def epanechikov(x):
    return torch.max(1 - torch.abs(x), torch.zeros_like(x))
```

<div align="center">
    <img src="./results/kernels.png" alt="different kernels">
    <p>Figure 1: Plot of different kernels</p>
</div>

Our implementation of the Nadaraya-Watson regression algorithm is as follows:

```python
def nadaraya_watson_regression(x_train, y_train, x_val, kernel):
    diff = x_val.view(-1,1) - x_train.view(1,-1)
    weights = kernel(diff).type(torch.float32)
    weights /= torch.sum(weights, dim=1, keepdim=True)
    y_pred = (weights @ y_train.view(-1,1)).squeeze(1)
    return y_pred
```

<div align="center">
    <img src="./results/regression_kernels.png" alt="regression kernels">
    <p>Figure 2: Estimated regression function using different kernels</p>
</div>

We can also use the Gaussian kernel with different bandwidths(adjust $\sigma$) . The following code shows how to do this:

```python
def adaptive_gaussian_regression(x_train, y_train, x_val, sigma:float=1.0):
    diff = x_val.view(-1,1) - x_train.view(1,-1)
    weights = ( - diff**2 / (2*sigma**2)).softmax(dim=-1)
    y_pred = (weights @ y_train.view(-1,1)).squeeze(1)
    return y_pred
```

<div align="center">
    <img src="./results/adaptive_gaussian_regression.png" alt="adaptive gaussian regression">
    <p>Figure 4: Estimated regression function using different bandwidths for the Gaussian kernel</p>
</div>

## References

- [Attention Pooling: Nadaraya-Watson Kernel Regression](https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-pooling.html)
- [Kernel Regression](https://en.wikipedia.org/wiki/Kernel_regression)
