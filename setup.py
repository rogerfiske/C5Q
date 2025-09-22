"""
Setup script for C5Q package installation.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="c5q",
    version="0.1.0",
    author="C5Q Development Team",
    author_email="dev@c5quantum.com",
    description="C5 Quantum Modeling - Least-20 Prediction System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/c5quantum/c5q",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "gpu": [
            "torch[cuda]>=2.1.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipython>=8.14.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "c5q-eda=c5q.eda:main",
            "c5q-train=c5q.train:main",
            "c5q-eval=c5q.eval:main",
        ],
    },
    include_package_data=True,
    package_data={
        "c5q": ["configs/*.yaml"],
    },
    zip_safe=False,
)