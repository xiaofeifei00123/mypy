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
from geopy.distance import distance  # 根据经纬度计算两点距离

pass
def latlon2distance(da2):
    """将剖面数据的经纬度横坐标变为距离坐标
    根据经纬度计算两点之间的距离

    Args:
        da2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    # flnm='/mnt/zfm_18T/fengxiang/HeNan/Data/GWD/d03/gwd0/cross1.nc'
    # flnm='/mnt/zfm_18T/fengxiang/HeNan/Data/GWD/d03/gwd0/cross1.nc'
    # ds = xr.open_dataset(flnm)
    # ds1 = ds.sel(time='2021-07-20 08')
    # da = ds1['wa_cross']
    # da1 = da.interpolate_na(dim='vertical', method='linear',  fill_value="extrapolate")
    # da2 = da1.sel(vertical=2000, method='nearest')
    dd = da2.xy_loc
    def str_latlon(string):
        # d1 = dd.values[0]
        lat = float(string.split(',')[0])
        lon = float(string.split(',')[1])
        return lat, lon

    d2 = dd.values
    lat_list = []
    lon_list = []
    for i in d2:
        # print(i)
        lat, lon = str_latlon(i)
        lat_list.append(lat)
        lon_list.append(lon)

    dis_list = [0]
    di = 0
    for i in range(len(lat_list)-1):
        # print(i)
        lat1 = lat_list[i]
        lon1 = lon_list[i]
        loc1 = (lat1, lon1)
        lat2 = lat_list[i+1]
        lon2 = lon_list[i+1]
        loc2 = (lat2, lon2)
        dist = distance(loc1,loc2).km
        di = di+dist
        dis_list.append(di)
    dis_list
    dis_array = (np.array(dis_list)).round(1)
    dis_array
    da2 = da2.assign_coords({'distance':('cross_line_idx',dis_array)})
    da3 = da2.swap_dims({'cross_line_idx':'distance'})
    da3
    return da3