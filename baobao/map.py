#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
创建地图的库

要求：
    1. 方便的创建
        shp
        合适标签的
        合适大小的

-----------------------------------------
Time             :2021/10/31 10:55:28
Author           :Forxd
Version          :1.0
'''
# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader, natural_earth
import matplotlib as mpl
from matplotlib.path import Path
import matplotlib.pyplot as plt
from cartopy.mpl import geoaxes

import numpy as np
# %%

# %%
class Map():
    pass
    def __init__(self) -> None:
        pass
        self.path_province = '/mnt/zfm_18T/fengxiang/DATA/SHP/Province_shp/henan.shp'
        self.path_city = '/mnt/zfm_18T/fengxiang/DATA/SHP/shp_henan/henan.shp'

    def create_map(self, ax, map_dic):
        """为geoax添加底图、标签等属性

        Args:
            ax ([type]): [description]

        Example:
            fig = plt.figure(figsize=(12, 12), dpi=600)
            proj = ccrs.PlateCarree()  # 创建坐标系
            ax = fig.add_axes([0.1,0.1,0.85,0.85], projection=proj)
            ma = Map()

            map_dic = {
                'proj':ccrs.PlateCarree(),
                'extent':[110, 117, 31, 37],
                'extent_interval_lat':1,
                'extent_interval_lon':1,
            }
            ma.create_map(ax, map_dic)
        # TODO  目前是只设置了河南省的地形文件，其他的待设置和安排
        """

        proj = map_dic['proj']
        ax.set_extent(map_dic['extent'], crs=proj)
        # ax.add_feature(cfeature.COASTLINE.with_scale('110m')) 
        province = cfeature.ShapelyFeature(
            Reader(self.path_province).geometries(),
            proj,
            edgecolor='k',
            facecolor='none')
        city = cfeature.ShapelyFeature(
            Reader(self.path_city).geometries(),
            proj,
            edgecolor='k',
            facecolor='none')
        ax.add_feature(province, linewidth=2, zorder=2) # zorder 设置图层为2, 总是在最上面显示
        ax.add_feature(city, linewidth=0.5, zorder=2) # zorder 设置图层为2, 总是在最上面显示

        ## 绘制坐标标签
        ax.set_yticks(np.arange(map_dic['extent'][2], map_dic['extent'][3] + map_dic['extent_interval_lat'], map_dic['extent_interval_lat']), crs=proj)
        ax.set_xticks(np.arange(map_dic['extent'][0], map_dic['extent'][1] + map_dic['extent_interval_lon'], map_dic['extent_interval_lon']), crs=proj)
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
        ax.tick_params(which='major',length=10,width=2.0) # 控制标签大小 
        ax.tick_params(which='minor',length=5,width=1.0)  #,colors='b')
        ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
        ax.tick_params(axis='both', labelsize=30, direction='out')
        return ax

    def add_station(self, ax, station, **note):
        """标记出站点和站点名

        Args:
            ax ([type]): [description]
        
        Example:
        
            station = {
                'ZhengZhou': {
                    'abbreviation':'ZZ',
                    'lat': 34.76,
                    'lon': 113.65
                },
            }
            ma.add_station(ax, station, justice=True)
        """
        pass
        values = station.values()
        station_name = list(station.keys())
        station_name = []
        x = []
        y = []
        for i in values:
            y.append(float(i['lat']))
            x.append(float(i['lon']))
            station_name.append(i['abbreviation'])

        # 标记出站点
        ax.scatter(x,
                   y,
                #    color='black',
                   color='black',
                   transform=ccrs.PlateCarree(),
                   alpha=1.,
                   linewidth=5,
                   s=35,
                   zorder=2
                   )
        # 给站点加注释
        if note['justice']:
            for i in range(len(x)):
                # print(x[i])
                ax.text(x[i]-0.2,
                        y[i] + 0.2,
                        station_name[i],
                        transform=ccrs.PlateCarree(),
                        alpha=1.,
                        fontdict={ 'size': 28, },
                        zorder=2,
                        )


if __name__ == '__main__':
############# 测试 ############
    fig = plt.figure(figsize=(12, 12), dpi=600)
    proj = ccrs.PlateCarree()  # 创建坐标系
    ax = fig.add_axes([0.1,0.1,0.85,0.85], projection=proj)
    ma = Map()

    map_dic = {
        'proj':ccrs.PlateCarree(),
        'extent':[110, 117, 31, 37],
        'extent_interval_lat':1,
        'extent_interval_lon':1,
    }
    ax = ma.create_map(ax, map_dic)

    station = {
        'ZhengZhou': {
            'abbreviation':'ZZ',
            'lat': 34.76,
            'lon': 113.65
        },
    }
    ma.add_station(ax, station, justice=False)


# %%
