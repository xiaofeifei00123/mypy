#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
一些常用的筛选数据的函数
-----------------------------------------
Time             :2022/01/06 20:14:40
Author           :Forxd
Version          :1.0
'''
import xarray as xr
import numpy as np

def select_area_latlon(da, area = {'lon1':112, 'lon2':113, 'lat1':33, 'lat2':34, 'interval':0.05}):
    """筛选经纬度范围内的数据
    经纬度的值可能不是规范的值，有时候是34.9999999, 就是不是35.0
    数字的小数位什么的也要注意

    Args:
        da ([type]): xarray.DataArray( lon: 20, lat: 20)
        area = {
            'lon1':112,
            'lon2':113,
            'lat1':33,
            'lat2':34,
            'interval':0.05,
        }

    Returns:
        [type]: [description]
    """
    da1 = xr.where((da.lat.round(2)<=area['lat2'])&(da.lat.round(2)>=area['lat1']) , da, np.nan)
    da2 = da1.dropna(dim='lat')
    da3 = xr.where((da2.lon.round(2)<=area['lon2'])&(da2.lon.round(2)>=area['lon1']) , da2, np.nan)
    da4 = da3.dropna(dim='lon')
    # da4
    return da4

