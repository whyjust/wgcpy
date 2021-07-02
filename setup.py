#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Project:       wgcpy
@File Name:     setup.py
@Author:        weiguang
@Date:          2021/6/29
"""
from setuptools import setup, find_packages

with open(r"./requirements.txt") as f:
    requires = [i for i in f if not i.startswith("#")]

setup(
    name="wgcpy",
    version="1.0.0",
    author="weiguang",
    author_email="1677234597@qq.com",
    description="Data analysis and PMML model framework!",
    packages=find_packages(),
    zip_safe=True,
    install_requires=requires,
    url="https://github.com/whyjust/wgcpy"
)
