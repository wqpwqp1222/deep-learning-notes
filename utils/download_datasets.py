import os

import torchvision.datasets as datasets

root = os.getenv('DNNL_DATA_ROOT', os.path.expanduser('~/datasets'))

try:
    ds = datasets.MNIST(root, download=True)
except Exception as err:
    raise ConnectionRefusedError(f'Error downloading MNIST dataset: {err}')

try:
    ds = datasets.Caltech101(root, download=True)
except Exception as err:
    raise ConnectionRefusedError(f'Error downloading Caltech101 dataset: {err}')
