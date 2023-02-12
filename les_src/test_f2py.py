#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
测试fortran
-----------------------------------------
Time             :2022/05/11 22:43:27
Author           :Forxd
Version          :1.0
'''
#%%
import numpy as np
import pymodule as py
from functools import wraps
from baobao.timedur import timeit
import time

#%%


n = 1000
a = np.empty(n, dtype=np.float32)
@timeit
def add(a,b):
    time.sleep(1)
    a = a+b
    return None

# py.fib(a, n)
# print(a)
add(100,200)
# add.timing.show_timing_log()


