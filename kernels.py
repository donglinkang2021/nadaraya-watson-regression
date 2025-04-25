import torch
import matplotlib.pyplot as plt
from pathlib import Path
Path('results').mkdir(parents=True, exist_ok=True)

# Define some kernels
def gaussian(x, sigma = 1.0):
    return torch.exp(-x**2 / (2*sigma**2))

def boxcar(x):
    return torch.abs(x) < 1.0

def constant(x):
    return 1.0 + 0 * x

def epanechikov(x):
    return torch.max(1 - torch.abs(x), torch.zeros_like(x))


def draw_different_kernel_func():
    x = torch.linspace(-3, 3, 100)
    kernels = [gaussian, boxcar, constant, epanechikov]
    names = ['Gaussian ($\sigma=1.0$)', 'Boxcar', 'Constant', 'Epanechikov']

    fig, axs = plt.subplots(1, len(kernels), figsize=(12, 3), sharey=True)
    for i, kernel in enumerate(kernels):
        ax:plt.Axes = axs[i]
        ax.plot(x.numpy(), kernel(x).numpy())
        ax.set_title(names[i])
        ax.set_xlabel('x')
        if i == 0:
            ax.set_ylabel('y')
    plt.suptitle('Plots of Different Kernels', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/kernels.png', dpi=300)
    plt.savefig('results/kernels.pdf')

def draw_different_sigma():
    x = torch.linspace(-3, 3, 100)
    sigmas = [0.1, 0.2, 0.5, 1.0]
    fig, axs = plt.subplots(1, len(sigmas), figsize=(12, 3), sharey=True)
    for i, sigma in enumerate(sigmas):
        ax:plt.Axes = axs[i]
        ax.plot(x.numpy(), gaussian(x, sigma).numpy())
        ax.set_title(f'Gaussian ($\sigma={sigma}$)')
        ax.set_xlabel('x')
        if i == 0:
            ax.set_ylabel('y')
    plt.suptitle('Gaussian Kernel with Different Sigma', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/gaussian_kernels.png', dpi=300)
    plt.savefig('results/gaussian_kernels.pdf')

if __name__ == "__main__":
    # Test the kernels
    draw_different_kernel_func()
    draw_different_sigma()