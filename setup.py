from setuptools import setup, find_packages

setup(
    name="copick-torch",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pytorch-lightning",
        "monai",
        "dask",
        "einops",
        "h5py",
        "magicgui",
        "numpy<2",
        "qtpy",
        "rich",
        "scikit-image",
        "scipy",
        "tensorboard",
        "mrcfile",
        "morphospaces @ git+https://github.com/kephale/morphospaces.git@copick",
        "copick @ git+https://github.com/copick/copick.git",
    ],
)
