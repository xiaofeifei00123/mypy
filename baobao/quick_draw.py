#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
快速绘图, 绘制带有色标和中国地图的图形
-----------------------------------------
Time             :2021/11/02 10:53:30
Author           :Forxd
Version          :1.0
'''

import cartopy.crs as ccrs
from nmc_met_graphics.plot import mapview
import matplotlib.pyplot as plt
import cmaps
import xarray as xr

from nmc_met_graphics.plot import mapview


def draw_contourf_single(data, ax):
    """画填色图
    """

    x = data.lon
    y = data.lat
    crx = ax.contourf(x,
                        y,
                        data,
                        corner_mask=False,
                        cmap = cmaps.precip3_16lev,
                        transform=ccrs.PlateCarree())
    return crx

def quick_contourf(da):
    mb = mapview.BaseMap()
    mb.drawcoastlines(linewidths=0.8, alpha=0.5)

    fig = plt.figure(figsize=[10,8])
    ax = fig.add_axes([0.12,0.03,0.8,0.97], projection=ccrs.PlateCarree())
    mb.drawstates(linewidths=0.8, alpha=0.5) # 省界
    mb.set_extent('中国陆地')
    cs = draw_contourf_single(da, ax)

    cb = fig.colorbar(
        cs,
        orientation='vertical',
        fraction=0.035,  # 色标大小
        pad=0.02,  # colorbar和图之间的距离
    )

def quick_contourf_station(rain):
    """rain[lon, lat, data],离散格点的DataArray数据

    Args:
        rain ([type]): [description]
    Example:
    da = xr.open_dataarray('/mnt/zfm_18T/fengxiang/HeNan/Data/OBS/rain_station.nc')
    da.max()
    rain = da.sel(time=slice('2021-07-20 00', '2021-07-20 12')).sum(dim='time')
    """
    mb = mapview.BaseMap()
    fig = plt.figure(figsize=[12, 12])
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=ccrs.LambertConformal(central_latitude=34, central_longitude=113))
    ax.plot(rain.lon.values, rain.lat.values, 'ko',ms=3,zorder=2, transform=ccrs.PlateCarree())

    colorlevel=[0, 0.1, 5, 15.0, 30, 70, 140, 700]#雨量等级
    colordict=['#F0F0F0','#A6F28F','#3DBA3D','#61BBFF','#0000FF','#FA00FA','#800040', '#EE0000',]#颜色列表
    cs = ax.tricontour(rain.lon, rain.lat, rain, levels=colorlevel, transform=ccrs.PlateCarree())
    cs = ax.tricontourf(rain.lon, rain.lat, rain, levels=colorlevel,colors=colordict, transform=ccrs.PlateCarree())
    fig.colorbar(cs)
    mb.drawstates(linewidths=0.8, alpha=0.5, zorder=2) # 省界
    mb.set_extent(region='河南')
    mb.cities(ax, city_type='base_station', color_style='black', 
                marker_size=5, font_size=16)
    mb.gridlines()