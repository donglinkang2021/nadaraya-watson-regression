import torch
import matplotlib.pyplot as plt
from pathlib import Path
from kernels import gaussian, boxcar, constant, epanechikov
Path('results').mkdir(parents=True, exist_ok=True)
torch.manual_seed(1337)

def f(x):
    return 2 * torch.sin(x) + x**0.8

def nadaraya_watson_regression(x_train, y_train, x_val, kernel):
    diff = x_val.view(-1,1) - x_train.view(1,-1)
    weights = kernel(diff).type(torch.float32)
    weights /= torch.sum(weights, dim=1, keepdim=True) # normalize weights
    y_pred = (weights @ y_train.view(-1,1)).squeeze(1)
    return y_pred

def test_different_kernels(x_train, y_train, x_val, y_val):
    kernels = [gaussian, boxcar, constant, epanechikov]
    names = ['Gaussian', 'Boxcar', 'Constant', 'Epanechikov']
    fig, axes = plt.subplots(1, len(kernels), figsize=(16, 5), sharey=True)
    axes = axes.flatten()
    for i, kernel in enumerate(kernels):
        y_pred = nadaraya_watson_regression(x_train, y_train, x_val, kernel=kernel)
        ax:plt.Axes = axes[i]
        ax.scatter(x_train, y_train, label='train data', s=20)
        ax.plot(x_val, y_val, label='true function', color='red')
        ax.plot(x_val, y_pred, label='prediction', color='green', linestyle='--')
        ax.set_title(f'{names[i]}')
        ax.set_xlabel('x')
        if i == 0:
            ax.set_ylabel('y')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, fontsize=14)
    plt.suptitle('Nadaraya-Watson Regression with Different Kernels', fontsize=16)
    # rect=[left, bottom, right, top] to adjust the layout
    plt.tight_layout(rect=[0, 0.08, 1, 1]) 
    plt.savefig('results/regression_kernels.png', dpi=300)
    plt.savefig('results/regression_kernels.pdf')
    plt.close(fig) # Close the figure to free memory

def test_adaptive_gaussian_regression(x_train, y_train, x_val, y_val):
    sigmas = [0.1, 0.2, 0.5, 1.0]
    fig, axes = plt.subplots(1, len(sigmas), figsize=(16, 5), sharey=True)
    axes = axes.flatten()
    for i, sigma in enumerate(sigmas):
        y_pred = nadaraya_watson_regression(x_train, y_train, x_val, kernel = lambda x: gaussian(x, sigma))
        ax:plt.Axes = axes[i]
        ax.scatter(x_train, y_train, label='train data', s=20)
        ax.plot(x_val, y_val, label='true function', color='red')
        ax.plot(x_val, y_pred, label='prediction', color='green', linestyle='--')
        ax.set_title(f'Gaussian ($\sigma={sigmas[i]}$)')
        ax.set_xlabel('x')
        if i == 0:
            ax.set_ylabel('y')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, fontsize=14)
    plt.suptitle('Adaptive Gaussian Regression with Different Sigma Values', fontsize=16)
    # rect=[left, bottom, right, top] to adjust the layout
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig('results/adaptive_gaussian_regression.png', dpi=300)
    plt.savefig('results/adaptive_gaussian_regression.pdf')
    plt.close(fig) # Close the figure to free memory

def draw_attention_heatmap(x_train, x_val):
    diff = x_val.view(-1,1) - x_train.view(1,-1)
    def gaussian_kernel(sigma):
        return lambda x: gaussian(x, sigma)
    kernels = [None, boxcar, constant, epanechikov, gaussian_kernel(0.1), gaussian_kernel(0.2), gaussian_kernel(0.5), gaussian_kernel(1.0)]
    names = [
        'Origin absolute difference',
        'Boxcar', 'Constant', 'Epanechikov', 
        'Gaussian (0.1)', 'Gaussian (0.2)', 'Gaussian (0.5)', 'Gaussian (1.0)'
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 10), sharey=True, sharex=True)
    axes = axes.flatten()
    for i, kernel in enumerate(kernels):
        if kernel is None:
            weights = torch.abs(diff).type(torch.float32) # origin kernel
        else:
            weights = kernel(diff).type(torch.float32)
        weights /= torch.sum(weights, dim=1, keepdim=True) # normalize weights
        ax:plt.Axes = axes[i]
        ax.imshow(
            weights.detach().numpy(), 
            cmap='Reds'
        )
        ax.set_title(f'{names[i]}')
        # ax.set_xlabel('x_train')
        # ax.set_ylabel('x_val')
        # ax.set_xticks([0, len(x_train)//2, len(x_train)-1])
        # ax.set_yticks([0, len(x_val)//2, len(x_val)-1])
        if i == 0 or i == 4:
            ax.set_ylabel('x_val')
        if i >= 4:
            ax.set_xlabel('x_train')
    plt.suptitle('Attention Heatmap of Different Kernels', fontsize=16)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig('results/attention_heatmap.png', dpi=300)
    plt.savefig('results/attention_heatmap.pdf')
    plt.close(fig) # Close the figure to free memory

if __name__ == '__main__':
    n_train = 50
    x_train, _ = torch.sort(torch.rand(n_train) * 5)
    y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))
    x_val = torch.arange(0, 5, 0.1)
    y_val = f(x_val)
    test_different_kernels(x_train, y_train, x_val, y_val)
    test_adaptive_gaussian_regression(x_train, y_train, x_val, y_val)
    draw_attention_heatmap(x_train, x_val)
