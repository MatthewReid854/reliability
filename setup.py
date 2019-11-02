#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reliability",
    version="0.3.4",
    description="Reliability Engineering toolkit for Python",
    author="Matthew Reid",
    author_email="m.reid854@gmail.com",
    license="MIT",
    url="https://github.com/MatthewReid854/reliability",
    keywords=["reliability","engineering","RAM","weibull","survival","analysis","censored","data","lifelines","probability","distributions","quality"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
    ],
    install_requires=["autograd>=1.3",
                      "scipy>=1.3.1",
                      "numpy>=1.17.1",
                      "matplotlib>=3.1.1",
                      "pandas>=0.23.4",
                      "autograd-gamma>=0.4.1",
    ],
    packages=setuptools.find_packages(),
)
