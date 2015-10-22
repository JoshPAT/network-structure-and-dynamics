#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Quan zhou'

import numpy as np
import os, functools

a = np.array([[1, 3],[2,3],[4,5]])

print a[-1:]

b = np.array([1, 3])

filepath = os.path.split(os.path.abspath(__file__))[0]
dirpath = os.path.join(filepath, 'datasets/')


def mkdir(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        # Create a dir
        global dirpath
        dir_name = func.__name__.split('_')[0]
        dirpath = os.path.join('datasets/', dir_name)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        result = func(*args, **kw)
        return result
    return wrapper


class Test():
    @staticmethod
    @mkdir
    def ttt():
        print os.path.exists(dirpath)

Test.ttt()

