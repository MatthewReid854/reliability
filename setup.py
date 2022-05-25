#!/usr/bin/env python

import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
    name="reliability",
    version="0.8.5",
    description="Reliability Engineering toolkit for Python",
    author="Matthew Reid",
    author_email="alpha.reliability@gmail.com",
    license="LGPLv3",
    url="https://reliability.readthedocs.io/en/latest/",
    project_urls={
        'Documentation': 'https://reliability.readthedocs.io/en/latest/',
        'Source Code': 'https://github.com/MatthewReid854/reliability',
        'Donate': 'https://reliability.readthedocs.io/en/latest/How%20to%20donate%20to%20the%20project.html'
    },
    keywords=[
        "reliability",
        "engineering",
        "RAM",
        "weibull",
        "lognormal",
        "exponential",
        "beta",
        "gamma",
        "normal",
        "loglogistic",
        "gumbel",
        "extreme",
        "value",
        "kaplan meier",
        "kaplan-meier",
        "survival",
        "analysis",
        "censored",
        "data",
        "lifelines",
        "probability",
        "distribution",
        "distributions",
        "fit",
        "fitting",
        "curve",
        "quality",
        "ALT",
        "accelerated",
        "life",
        "testing",
        "MCF",
        "mean",
        "cumulative",
        "CIF",
        "DS",
        "ZI",
        "defective",
        "subpopulation",
        "zero",
        "inflated",
        "DSZI",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    ],
    install_requires=[
        "autograd>=1.3",
        "scipy>=1.7.0",
        "numpy>=1.19.2",
        "matplotlib>=3.5.0",
        "pandas>=1.1.2",
        "autograd-gamma>=0.5.0",
        "mplcursors>=0.3",
        "docutils<0.18"
    ],
    python_requires=">=3.7",
    packages=setuptools.find_packages(
        exclude=["*.tests", "*.tests.*"]
    ),
)
