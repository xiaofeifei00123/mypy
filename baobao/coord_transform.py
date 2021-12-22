#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
坐标转换
新增加x,y个数不等的处理方法
利用numpy的meshgrid, 给每个格点创建了经纬度
-----------------------------------------
Time             :2021/11/04 17:31:03
Author           :Forxd
Version          :2.0
'''

import netCDF4 as nc
import xarray as xr
import wrf
import numpy as np


pass