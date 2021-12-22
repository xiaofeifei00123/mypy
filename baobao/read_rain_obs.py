#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
读取观测站点降水，并插值
观测站降水为0时，该时次观测站数据没有显示
需要对这些数据进行添加
保存：
    插值的数据，nc搁谁
    未插值的数据, csv
# TODO, 格点插值的程序未做完
-----------------------------------------
Time             :2021/09/28 20:39:34
Author           :Forxd
Version          :1.0
'''

# %%
# from HeNan.Draw.read_ERA5 import get_rain_station
import xarray as xr
import meteva.base as meb
import numpy as np
import os
import pandas as pd
from metpy.interpolate import inverse_distance_to_grid
from metpy.interpolate import interpolate_to_grid
from baobao.quick_draw import quick_contourf,quick_contourf_station
# %%


# %%
class RainStation():
    """站点数据聚合到一起，并未没有数据的站点赋值为0"""
    def read_one_station(self, flnm):
        station = meb.read_stadata_from_micaps3(flnm)
        ## 转换为世界时
        station['time'] = station['time']+pd.Timedelta('-8H')

        df = station

        ## 将一个站点的数据，变为DataArray
        da = xr.DataArray(
            df['data0'].values,
            coords={
                'id':df['id'],
                'lat':('id',df['lat']),
                'lon':('id',df['lon']),
                'time':df['time'].values[0],
            },
            dims=['id',]
        )
        return da

    def get_rain_station(self,):
        path='/mnt/zfm_18T/fengxiang/HeNan/Data/OBS/obs_station'
        fl_list = os.popen('ls {}/2021*.000'.format(path))  # 打开一个管道
        fl_list = fl_list.read().split()
        dds_list = []
        for fl in fl_list:
            da = self.read_one_station(fl)
            dds_list.append(da)

        ## 针对micaps数据的各个站点数据进行聚合
        da_concat = xr.concat(dds_list, dim='time')
        lat = da_concat['lat'].mean(dim='time')  # 将多列数据，变成一列
        lon = da_concat['lon'].mean(dim='time')
        dda = da_concat.drop_vars(['lat', 'lon'])
        daa = dda.assign_coords({'lat':('id',lat.values), 'lon':('id',lon.values)})
        dc = daa.fillna(0)
        return dc
        
    def save_rain_station(self,):
        ## 保存成micaps3格式, 保存站点数据
        # rs = rain_station()
        rain_st = self.get_rain_station()
        rain_st.to_netcdf('/mnt/zfm_18T/fengxiang/HeNan/Data/OBS/rain_station.nc')
        
class RainGrid():
    """格点数据

    Returns:
        [type]: [description]
    """
    pass
    def rain_station2grid(self, da):
        """将站点数据，插值为格点数据

        Args:
            da ([type]): [description]

        Returns:
            [type]: [description]
        """
        # interval = 0.125

        
        ## 这个area是有头有尾， 有中间数据的
        ## +1， -1是设置一个更大范围, 保证在想要范围内的数据一致性
        area = {
            'lon1':110,
            'lon2':116,
            'lat1':31,
            'lat2':37,
            'interval':0.125,
        }

        lon_grid = np.arange(area['lon1'], area['lon2']+area['interval'], area['interval'])
        lat_grid = np.arange(area['lat1'], area['lat2']+area['interval'], area['interval'])
        
        
        # lon_grid = np.arange(111-1, 115+1+interval, interval)
        # lat_grid = np.arange(32-1, 36+1+interval, interval)
        lon_gridmesh, lat_gridmesh = np.meshgrid(lon_grid, lat_grid)

        grid_list = []
        tt = da.time

        for i in tt:
            print('插值%s'%(i.dt.strftime('%Y-%m-%d %H').values))
            da_hour = da.sel(time=i)
            hour_grid = inverse_distance_to_grid(da.lon.values, da.lat.values, da_hour.values, lon_gridmesh, lat_gridmesh, 
                                                 r=0.5, min_neighbors=3,
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

    def save_rain_grid(self,):
        ## 读取存储好的站点数据
        da = xr.open_dataarray('/mnt/zfm_18T/fengxiang/HeNan/Data/OBS/rain_station.nc')
        ## 插值为格点数据
        da_grid = self.rain_station2grid(da)
        da_grid.to_netcdf('/mnt/zfm_18T/fengxiang/HeNan/Data/OBS/rain_grid.nc')

def save_rain():
    # rs = RainStation()
    # rs.save_rain_station()
    rg = RainGrid()
    rg.save_rain_grid()
            
        

# %%
if __name__ == '__main__':
    pass
    save_rain()
    # save_rain_grid()
    # save_rain_station()
    
