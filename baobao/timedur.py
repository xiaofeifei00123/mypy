#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
和时间或者和装饰器相关的函数
-----------------------------------------
Author           :Forxd
Version          :1.0
Time：2022/05/11/ 23:34
'''
import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def inner(*args, **kwargs):  # 使用 *args, **kwargs 适应所有参数
        start = time.time()
        res = func(*args, **kwargs)    # 传递参数给真实调用的方法
        end = time.time()
        durt = format((end-start),'0.3f')
        print('duration time is {} s'.format(durt))
        return res
    return inner

