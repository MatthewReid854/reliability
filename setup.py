#!/usr/bin/env python

import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
    name="reliability",
    version="0.5.3",
    description="Reliability Engineering toolkit for Python",
    author="Matthew Reid",
    author_email="alpha.reliability@gmail.com",
    license="LGPLv3",
    url="https://reliability.readthedocs.io/en/latest/",
    keywords=["reliability","engineering","RAM","weibull","lognormal","exponential","beta","gamma","normal","loglogistic","kaplan meier","kaplan-meier","survival","analysis","censored","data","lifelines","probability","distribution","distributions","quality","ALT","accelerated","life","testing","MCF","mean","cumulative","CIF"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3"
    ],
    install_requires=["autograd>=1.3",
                      "scipy>=1.3.2",
                      "numpy>=1.17.1",
                      "matplotlib>=3.1.1",
                      "pandas>=0.23.4",
                      "autograd-gamma>=0.4.1",
                      "mplcursors>=0.3"
    ],
    packages=setuptools.find_packages(exclude=["*.tests", "*.tests.*","*.notebooks", "*.notebooks.*"]),
)