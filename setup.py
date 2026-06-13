"""Package installer for MTSC Sulci unsupervised training."""

from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).parent
README = ROOT / "README.md"


setup(
    name="MTSC-sulci",
    version="0.1.0",
    description="3D MRI sulcal pre-training, fine-tuning, and classification workflows.",
    long_description=README.read_text(encoding="utf-8") if README.exists() else "",
    long_description_content_type="text/markdown",
    author="Michail Mamalakis",
    license="MIT",
    python_requires=">=3.10",
    packages=[
        "MTSC_sulci",
        "MTSC_sulci.fine_tuning",
        "MTSC_sulci.pre_training",
        "MTSC_sulci.scripts",
        "MTSC_sulci.utilities",
    ],
    package_dir={
        "MTSC_sulci": ".",
        "MTSC_sulci.fine_tuning": "fine_tuning",
        "MTSC_sulci.pre_training": "pre_training",
        "MTSC_sulci.scripts": "scripts",
        "MTSC_sulci.utilities": "utilities",
    },
    package_data={
        "MTSC_sulci": ["configs/*.yaml"],
    },
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "scikit-image",
        "nibabel",
        "torch",
        "torchvision",
        "monai",
        "monai-generative",
        "lightning",
        "wandb",
        "minlora",
        "tensorflow",
        "tf2onnx",
        "onnx",
        "onnx2pytorch",
        "PyYAML",
        "keras-resnet3d @ git+https://github.com/JihongJu/keras-resnet3d.git",
    ],
    extras_require={
        "xformers": ["xformers"],
        "test": [],
    },
    entry_points={
        "console_scripts": [
            "mtsc-pre-training=MTSC_sulci.scripts.pre_training:main",
            "mtsc-classification=MTSC_sulci.scripts.classification:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
