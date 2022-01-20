#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
插值函数的分装
-----------------------------------------
Author           :Forxd
Version          :1.0
Time：2021/11/03/ 10:48
'''

import xarray as xr
import xesmf as xe
import numpy as np
from metpy.interpolate import inverse_distance_to_grid
from metpy.interpolate import interpolate_to_grid


def regrid_xesmf(dataset, area, rd=1):
    """利用xESMF库，将非标准格点的数据，插值到标准格点上去
    注意：dataset的coords, lat,lon 必须同时是一维或是二维的
    Args:
        dataset ([type]): Dataset格式的数据, 多变量，多时次，多层次
        area, 需要插值的网格点范围, 即latlon坐标的经纬度范围
        rd: 数据保留的小数位
    """
    ## 创建ds_out, 利用函数创建,这个东西相当于掩膜一样
    ds_regrid = xe.util.grid_2d(area['lon1']-area['interval']/2, 
                                area['lon2'], 
                                area['interval'], 
                                area['lat1']-area['interval']/2, 
                                area['lat2'],
                                area['interval'])
    #ds_regrid = xe.util.grid_2d(area['lon1'], area['lon2'], area['interval'], area['lat1'], area['lat2'], area['interval'])
    # ds_regrid = xe.util.grid_2d(110, 116, 0.05, 32, 37, 0.05)
    regridder = xe.Regridder(dataset, ds_regrid, 'bilinear')  # 好像是创建了一个掩膜一样
    ds_out = regridder(dataset)  # 返回插值后的变量

    ### 重新构建经纬度坐标
    lat = ds_out.lat.sel(x=0).values.round(3)
    lon = ds_out.lon.sel(y=0).values.round(3)
    ds_1 = ds_out.drop_vars(['lat', 'lon']).round(rd)  # 可以删除variable和coords

    ## 设置和dims, x, y相互依存的coords, 即lat要和y的维度一样
    ds2 = ds_1.assign_coords({'lat':('y',lat), 'lon':('x',lon)})
    # ## 将新的lat,lon, coords设为dims
    ds3 = ds2.swap_dims({'y':'lat', 'x':'lon'})
    ## 删除不需要的coords
    # ds_return = ds3.drop_vars(['XTIME'])
    ds_return = ds3
    return ds_return


def rain_station2grid(da, area, r=0.2, min_neighbors=3):
    """将站点数据，插值为格点数据

    Args:
    da:
     da = xr.DataArray(
                da.values,
                coords = {
                    'id':id,
                    'lat':('id', lat),
                    'lon':('id', lon),
                    'time':time
                },
                dims = ['id', 'time']
                )

    area:
        ## 这个area是有头有尾， 有中间数据的, [32, 32.125, ...., 37]
        area = {
            'lon1':110,
            'lon2':116,
            'lat1':31,
            'lat2':37,
            'interval':0.125,
        }

    Returns:
        [type]: [description]
    """

    lon_grid = np.arange(area['lon1'], area['lon2']+area['interval'], area['interval'])
    lat_grid = np.arange(area['lat1'], area['lat2']+area['interval'], area['interval'])
    lon_gridmesh, lat_gridmesh = np.meshgrid(lon_grid, lat_grid)

    grid_list = []
    tt = da.time

    for i in tt:
        print('插值%s'%(i.dt.strftime('%Y-%m-%d %H').values))
        da_hour = da.sel(time=i)
        hour_grid = inverse_distance_to_grid(da.lon.values, da.lat.values, da_hour.values, lon_gridmesh, lat_gridmesh, 
                                             r=r, min_neighbors=min_neighbors,
                                             )

        da_hour_grid = xr.DataArray(hour_grid.round(1),
                                    coords={
                                        'lon':lon_grid,
                                        'lat':lat_grid,
                                        'time':i
                                    },
                                    dims=['lat', 'lon'])
        grid_list.append(da_hour_grid)
    ddc = xr.concat(grid_list,dim='time')
    return ddc

# def grid2station(cc):
#     """格点数据插值到站点上
#     """
#     sta = cc.id.values
#     lon = xr.DataArray(cc.lon.values, coords=[sta], dims=['sta'])
#     lat = xr.DatArray(cc.lat.values, coords=[sta], dims=['sta'])
#     rr = rain.interp(lon=lon, lat=lat, method='nearest').round(1)
