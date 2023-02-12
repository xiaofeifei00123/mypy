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
import matplotlib.ticker as ticker
from cartopy.mpl import geoaxes
import pandas as pd

import numpy as np
# %%

# %%
class Map():
    pass
    def __init__(self) -> None:
        pass
        self.path_china='/mnt/zfm_18T/fengxiang/DATA/SHP/Map/cn_shp/Province_9/Province_9.shp'
        # self.path_china= '/mnt/zfm_18T/fengxiang/DATA/SHP/shp_micaps/continents_lines.shp'
        # self.path_china= '/mnt/zfm_18T/fengxiang/DATA/SHP/shp_micaps/NationalBorder.shp'
        # self.path_china= '/mnt/zfm_18T/fengxiang/DATA/SHP/shp_micaps/County.shp'
        self.path_province = '/mnt/zfm_18T/fengxiang/DATA/SHP/Province_shp/henan.shp'
        self.path_city = '/mnt/zfm_18T/fengxiang/DATA/SHP/shp_henan/henan.shp'

    def create_map(self, ax, map_dic, ):
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
        # ax.set_ylim(map_dic['extent'][2], map_dic['extent'][3])
        # ax.set_xlim(map_dic['extent'][0], map_dic['extent'][1])
        # ax.add_feature(cfeature.COASTLINE.with_scale('110m')) 
        country = cfeature.ShapelyFeature(
            Reader(self.path_china).geometries(),
            proj,
            edgecolor='k',
            facecolor='none')
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
        # ax.add_feature(province, linewidth=1, zorder=2, alpha=0.6) # zorder 设置图层为2, 总是在最上面显示
        ax.add_feature(city, linewidth=0.5, zorder=2, alpha=0.5) # zorder 设置图层为2, 总是在最上面显示
        ax.add_feature(country, linewidth=1, zorder=2, alpha=0.5) # zorder 设置图层为2, 总是在最上面显示

        ## 绘制坐标标签
        ax.set_yticks(np.arange(map_dic['extent'][2], map_dic['extent'][3] + map_dic['extent_interval_lat'], map_dic['extent_interval_lat'], dtype='int'), crs=proj)
        ax.set_xticks(np.arange(map_dic['extent'][0], map_dic['extent'][1] + map_dic['extent_interval_lon'], map_dic['extent_interval_lon'], dtype='int',), crs=proj)

        # ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        # ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

        ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol="$^{\circ}$"))  # 使用半角的度，用latex公式给出
        ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol="$^{\circ}$"))

        # ax.tick_params(axis='both', labelsize=10, direction='out')
        # ax.tick_params(which='major',length=6,width=1.0) # 控制标签大小 
        # ax.tick_params(which='minor',length=3,width=0.5)  #,colors='b')
        ax.tick_params(axis='both', labelsize=8, direction='out')
        ax.tick_params(which='major',length=4,width=0.8) # 控制标签大小 
        ax.tick_params(which='minor',length=2,width=0.4)  #,colors='b')

        return ax

    def create_map_china(self, ax, 
            map_dic = {
                'proj':ccrs.PlateCarree(),
                # 'extent':[110, 117, 31, 37],
                'extent':[108, 120, 30, 38],
                'extent_interval_lat':1,
                'extent_interval_lon':1,
            },):
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
            Reader(self.path_china).geometries(),
            proj,
            edgecolor='k',
            facecolor='none')
        city = cfeature.ShapelyFeature(
            Reader(self.path_city).geometries(),
            proj,
            edgecolor='k',
            facecolor='none')
        ax.add_feature(province, linewidth=1, zorder=2, alpha=0.6) # zorder 设置图层为2, 总是在最上面显示
        # ax.add_feature(city, linewidth=0.5, zorder=2, alpha=0.5) # zorder 设置图层为2, 总是在最上面显示

        ## 绘制坐标标签
        ax.set_yticks(np.arange(map_dic['extent'][2], map_dic['extent'][3] + map_dic['extent_interval_lat'], map_dic['extent_interval_lat'], dtype='int'), crs=proj)
        ax.set_xticks(np.arange(map_dic['extent'][0], map_dic['extent'][1] + map_dic['extent_interval_lon'], map_dic['extent_interval_lon'], dtype='int',), crs=proj)

        # ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        # ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

        ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol="$^{\circ}$"))  # 使用半角的度，用latex公式给出
        ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol="$^{\circ}$"))

        # ax.tick_params(axis='both', labelsize=10, direction='out')
        # ax.tick_params(which='major',length=6,width=1.0) # 控制标签大小 
        # ax.tick_params(which='minor',length=3,width=0.5)  #,colors='b')
        ax.tick_params(axis='both', labelsize=8, direction='out')
        ax.tick_params(which='major',length=4,width=0.8) # 控制标签大小 
        ax.tick_params(which='minor',length=2,width=0.4)  #,colors='b')

        return ax

    def add_station(self, ax, station, **note):
        """标记出站点和站点名

        Args:
            ax ([type]): [description]
            note: 相当于传了个字典进来了，如果没有参数的话，就是空的字典
            参考：https://www.cnblogs.com/bingabcd/p/6671368.html
        
        Example:
        
            station = {
                'ZhengZhou': {
                    'abbreviation':'ZZ',
                    'lat': 34.76,
                    'lon': 113.65
                },
            }
            ma.add_station(ax, station, justice=True, delx=0.1)
        """
        pass
        fontsize = 10
        ssize = 12
        marker = '.'
        if 'fontsize' in note.keys():
            fontsize = note['fontsize']
        if 'ssize' in note.keys():
            ssize = note['ssize']
        if 'marker' in note.keys():
            marker = note['marker']



        values = station.values()
        # station_name = list(station.keys())
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
                #    linewidth=5,
                   s=ssize,
                   zorder=2,
                   marker=marker,
                   )
        # 给站点加注释
        # if note:
        #     pass
        if 'justice' in note.keys():
            if note['justice']:
                for i in range(len(x)):
                    # print(x[i])
                    delx = -0.2
                    dely = 0.1
                    if 'delx' in note.keys():
                        delx = note['delx']
                    if 'dely' in note.keys():
                        dely = note['dely']
                    ax.text(x[i] + delx,
                            y[i] + dely,
                            station_name[i],
                            transform=ccrs.PlateCarree(),
                            alpha=1.,
                            fontdict={ 'size': fontsize, },
                            zorder=2,
                            # bbox={'boxstyle': 'square', 'facecolor': 'white', edgecolor='white'},
                            )

def draw_south_sea(fig,):
    pass
    ax2 = fig.add_axes([0.798, 0.145, 0.2, 0.2],projection=ccrs.PlateCarree())
    ax2.add_geometries(Reader('/mnt/zfm_18T/fengxiang/DATA/SHP/Map/cn_shp/Province_9/Province_9.shp').geometries(),ccrs.PlateCarree(),facecolor='none',edgecolor='black',linewidth=0.8)


def get_rgb(fn):
    """
    fn: rgb_txt文件存储路径
    """
    # fn = './11colors.txt'
    df = pd.read_csv(fn, skiprows=4, sep='\s+',encoding='gbk',header=None, names=['r','g','b'])
    rgb = []
    for ind, row in df.iterrows():
        rgb.append(row.tolist())
    rgb = np.array(rgb)/255.
    return rgb



if __name__ == '__main__':
############# 测试 ############
    # fig = plt.figure(figsize=(12, 12), dpi=600)
    fig = plt.figure(figsize=(12, 8), dpi=600)
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
