"""Setup script for the gym_2048 package."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="envpack",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "absl-py>=2.3.1",
        "gymnasium>=1.2.3",
        "numpy>=2.3.5",
        "pillow>=11.3.0",
        "matplotlib>=3.10.8",
        "numba>=0.63.1",
    ],
    author="Rakesh Iyer",
    author_email="your.email@example.com",
    description="A collection of game environments for Gymnasium.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rax85/envpack",
    license="Apache-2.0",
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
)
