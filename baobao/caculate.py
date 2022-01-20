#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
分装一些计算诊断变量的函数

-----------------------------------------
Author           :Forxd
Version          :1.0
Time：2021/11/03/ 10:33
'''

import numpy as np
import xarray as xr
import pandas as pd
from metpy import calc as ca  # calc是一个文件夹
from metpy.units import units  # units是units模块中的一个变量名(函数名？类名？)
from metpy import constants  # constatns是一个模块名
from metpy.calc import specific_humidity_from_dewpoint
from metpy.calc import mixing_ratio_from_specific_humidity
from metpy.calc import virtual_potential_temperature
from metpy.calc import potential_temperature
from metpy.calc import relative_humidity_from_dewpoint
from metpy.calc import dewpoint_from_relative_humidity
from metpy.calc import equivalent_potential_temperature
import metpy.interpolate as interp


def caculate_q_rh_theta(ds):
    """计算比湿，位温等诊断变量
    theta, theta_e, theta_v
    主要用于计算探空资料获得的数据
    传入的温度和露点必须是摄氏温度
    根据td或者rh计算q,theta_v
    返回比湿q, 虚位温theta_v, 相对湿度rh

    Args:
        ds (Dataset): 包含有temp ,td的多维数据
        这里传入Dataset合理一点
    """
    pass        
    ## 获得温度和露点温度
    dims_origin = ds['temp'].dims  # 这是一个tuple, 初始维度顺序
    ds = ds.transpose(*(...,'pressure'))

    var_list = ds.data_vars
    t = ds['temp']

    ## 转换单位
    pressure = units.Quantity(t.pressure.values, "hPa")

    ## 针对给的rh 或是td做不同的计算
    ## 需要确定t的单位
    if 'td' in var_list:
        """探空资料的温度单位大多是degC"""
        td = ds['td']
        dew_point = units.Quantity(td.values, "degC")
        temperature = units.Quantity(t.values, "degC")
    elif 'rh' in var_list:
        """FNL资料的单位是K"""
        # rh = da.sel(variable='rh')
        rh = ds['rh']
        rh = units.Quantity(rh.values, "%")
        temperature = units.Quantity(t.values, "degC")
        dew_point = dewpoint_from_relative_humidity(temperature, rh)
    else:
        print("输入的DataArray中必须要有rh或者td中的一个")
    
    ## 记录维度坐标
    # time_coord = t.time.values
    # pressure_coord = t.pressure.values

    ## 计算诊断变量
    q = specific_humidity_from_dewpoint(pressure, dew_point)
    w = mixing_ratio_from_specific_humidity(q)
    theta_v = virtual_potential_temperature(pressure, temperature, w)

    theta = potential_temperature(pressure, temperature)
    theta_e = equivalent_potential_temperature(pressure, temperature, dew_point)

    if 'td' in var_list:
        rh = relative_humidity_from_dewpoint(temperature, dew_point)
        var_name_list = ['q', 'rh', 'theta_v', 'theta', 'theta_e']
        var_data_list = [q, rh, theta_v, theta, theta_e]
    elif 'rh' in var_list:
        pass
        var_name_list = ['q', 'td', 'theta_v', 'theta', 'theta_e']
        var_data_list = [q, dew_point, theta_v, theta, theta_e]

    ## 融合各物理量为一个DataArray

    ds_return = xr.Dataset()

    for var_name, var_data in zip(var_name_list, var_data_list):
        pass
        ## 为了去除单位
        dda = xr.DataArray(
            var_data, 
            # coords=[time_coord,pressure_coord],
            coords = t.coords,
            dims=t.dims)
            # dims=['time', 'pressure'])

        ds_return[var_name] = xr.DataArray(
            dda.values, 
            # coords=[time_coord,pressure_coord],
            coords=t.coords,
            dims=t.dims)
    ## 转换维度顺序        
    ds_return = ds_return.transpose(*dims_origin)
    return ds_return











def caculate_q_rh_thetav(ds):
    """计算比湿，位温等诊断变量
    用于计算wrfout所获得的数据
    传入的温度和露点必须是摄氏温度
    根据td或者rh计算q,theta_v
    返回比湿q, 虚位温theta_v, 相对湿度rh

    Args:
        ds (Dataset): 包含有temp ,td的多维数据
        这里传入Dataset合理一点
    """
    pass        
    ## 获得温度和露点温度
    dims_origin = ds['temp'].dims  # 这是一个tuple, 初始维度顺序
    ds = ds.transpose(*(...,'pressure'))

    var_list = ds.data_vars
    t = ds['temp']

    ## 转换单位
    pressure = units.Quantity(t.pressure.values, "hPa")

    ## 针对给的rh 或是td做不同的计算
    ## 需要确定t的单位
    if 'td' in var_list:
        """探空资料的温度单位大多是degC"""
        td = ds['td']
        dew_point = units.Quantity(td.values, "degC")
        temperature = units.Quantity(t.values, "degC")
    elif 'rh' in var_list:
        """FNL资料的单位是K"""
        # rh = da.sel(variable='rh')
        rh = ds['rh']
        rh = units.Quantity(rh.values, "%")
        temperature = units.Quantity(t.values, "degC")
        dew_point = dewpoint_from_relative_humidity(temperature, rh)
    else:
        print("输入的DataArray中必须要有rh或者td中的一个")
    
    ## 记录维度坐标
    # time_coord = t.time.values
    # pressure_coord = t.pressure.values

    ## 计算诊断变量
    q = specific_humidity_from_dewpoint(pressure, dew_point)
    w = mixing_ratio_from_specific_humidity(q)
    theta_v = virtual_potential_temperature(pressure, temperature, w)

    theta = potential_temperature(pressure, temperature)
    theta_e = equivalent_potential_temperature(pressure, temperature, dew_point)

    if 'td' in var_list:
        rh = relative_humidity_from_dewpoint(temperature, dew_point)
        # var_name_list = ['q', 'rh', 'theta_v', 'theta', 'theta_e']
        # var_data_list = [q, rh, theta_v, theta, theta_e]
        var_name_list = ['q', 'rh', 'theta_v']
        var_data_list = [q, rh, theta_v]
    elif 'rh' in var_list:
        pass
        # var_name_list = ['q', 'td', 'theta_v', 'theta', 'theta_e']
        # var_data_list = [q, dew_point, theta_v, theta, theta_e]
        var_name_list = ['q', 'td', 'theta_v']
        var_data_list = [q, dew_point, theta_v]

    ## 融合各物理量为一个DataArray

    ds_return = xr.Dataset()

    for var_name, var_data in zip(var_name_list, var_data_list):
        pass
        ## 为了去除单位
        dda = xr.DataArray(
            var_data, 
            # coords=[time_coord,pressure_coord],
            coords = t.coords,
            dims=t.dims)
            # dims=['time', 'pressure'])

        ds_return[var_name] = xr.DataArray(
            dda.values, 
            # coords=[time_coord,pressure_coord],
            coords=t.coords,
            dims=t.dims)
    ## 转换维度顺序        
    ds_return = ds_return.transpose(*dims_origin)
    return ds_return


class QvDiv():
    """水汽通量散度的计算,
    这里分装这个类的意图是使得计算水汽通量散度更好区分，
    每个函数需要传递的ds没有相关关系
    """
    def __init__(self) -> None:
        pass
    def caculate_qv_div(self, ds):
        """计算单层, 单个时次水汽通量散度

        Args:
            ds (Dataset): 
                lat, lon坐标的多维数组, 包含有q,u,v等变量
                q: kg/kg
                u: m/s
                v: m/s

        Returns:
            qv_div : 计算好的水汽通量散度
        """
        lon = ds.lon
        lat = ds.lat
        u = ds.u*units('m/s')
        v = ds.v*units('m/s')
        q = ds.q*10**3*units('g/kg')
        qv_u = q*u/constants.g
        qv_v = q*v/constants.g

        dx, dy = ca.lat_lon_grid_deltas(lon.values, lat.values)
        qv_div = ca.divergence(u=qv_u, v=qv_v, dx=dx, dy=dy)
        return qv_div

    # dds.sel(pressure=500)
    def caculate_qv_div_integral(self, dds):
        """计算单个时次整层水汽通量散度, 多个层次的积分值
        其实就是个算梯度积分的函数

        Args:
            dds (Dataset): 包含u,v,q(kg/kg)变量, 维度是lat, lon, pressure三维, 单个时次的

        Example:
            flnm_wrf = '/mnt/zfm_18T/fengxiang/HeNan/Data/high_resolution_high_hgt_upar_d04_latlon.nc'
            ds = xr.open_dataset(flnm_wrf)
            dds = ds.sel(pressure=[1000, 925, 850, 700, 500]).isel(time=0)
        """
        lev = dds.pressure
        qv_div_list = []
        for i in lev.values:
            ds_one = dds.sel(pressure=i)
            qv_div = self.caculate_qv_div(ds_one)
            qv_div_list.append(qv_div)
        qv_div_integral = np.trapz(qv_div_list[::-1], lev.values[::-1], axis=0)
        da_qv_div_integral = xr.DataArray(qv_div_integral, 
                        coords={
                            'lat':qv_div.lat.values,
                            'lon':qv_div.lon.values,
                        }, 
                        dims=['lat', 'lon'])
        return da_qv_div_integral


    def caculate_qv_div_time(self, ds):
        """计算某一层所有时次的水汽通量散度
        例如500hPa多个时次的水汽通量散度
        把不同时次的值聚合在一起而已

        Args:
            ds ([type]): [description]

        Returns:
            [type]: [description]
        """
        pass
        tt = ds.time
        qv_div_list = []
        for t in tt:
            qvd = self.caculate_qv_div(ds.sel(time=t))
            qv_div_list.append(qvd)
        qv_div_dual_time = xr.concat(qv_div_list, dim='time')
        return qv_div_dual_time
            
    def caculate_qv_div_integral_time(self, ds):
        """计算多个时次的整层水汽通量散度
        把不同时次的值聚合在一起而已

        Args:
            ds (Dataset): 多时次，多层次的, 含有变量q,v,u

        Returns:
            [type]: [description]
        """
        pass
        tt = ds.time
        qv_div_integral_list = []
        for t in tt:
            qvdi = self.caculate_qv_div_integral(ds.sel(time=t))
            qv_div_integral_list.append(qvdi)
        qv_div_dual_time = xr.concat(qv_div_integral_list, dim='time')
        return qv_div_dual_time

    def test(self,):
        """用来测试上述函数的准确性的
        """
        flnm_wrf = '/mnt/zfm_18T/fengxiang/HeNan/Data/high_resolution_high_hgt_upar_d04_latlon.nc'
        ds = xr.open_dataset(flnm_wrf)
        # dds = ds.sel(pressure=500).isel(time=0)
        # dds = ds.sel(pressure=500)
        dds = ds.sel(pressure=[1000, 925, 850, 700, 500])
        cc = self.caculate_qv_div_integral_time(ds)
        # da = self.caculate_qv_div_time(dds)
        print(cc.min())


class Qv():
    """水汽通量的计算,
    每个函数需要传递的ds没有相关关系, 只是都是Dataset这种格式,
    所以不能做到统一输入
    最原始的ds应该满足
    Dimensions:(lat, lon, pressure, time)
    variables:u, v, q
    """
    
    def __init__(self) -> None:
        pass

    def caculate_qv(self, ds):
        """计算单层, 单个时次水汽通量

        Args:
            ds (Dataset): 
                lat, lon坐标的多维数组, 包含有q,u,v等变量
                q: kg/kg
                u: m/s
                v: m/s

        Returns:
            qv_div : 计算好的水汽通量散度
        """
        u = ds.u*units('m/s')
        v = ds.v*units('m/s')
        q = ds.q*10**3*units('g/kg')
        qv_u = q*u/constants.g
        qv_v = q*v/constants.g
        # qf = xr.ufuncs.sqrt(qv_u**2+qv_v**2) # 水汽通量的大小,模
        qf = np.sqrt(qv_u**2+qv_v**2) # 水汽通量的大小,模
        dda = xr.concat([qv_u, qv_v, qf], pd.Index(['qv_u', 'qv_v', 'qv_f'], name='model'))
        dds = dda.to_dataset(dim='model')
        return dds


    def caculate_qv_integral(self, dds):
        """计算水汽通量整层积分

        Args:
            dds (Dataset): 包含u,v,q(kg/kg)变量, 维度是lat, lon, pressure三维, 单个时次的

        Example:
            flnm_wrf = '/mnt/zfm_18T/fengxiang/HeNan/Data/high_resolution_high_hgt_upar_d04_latlon.nc'
            ds = xr.open_dataset(flnm_wrf)
            dds = ds.sel(pressure=[1000, 925, 850, 700, 500]).isel(time=0)
        """
        lev = dds.pressure
        qv_list = []
        for i in lev.values:
            ds_one = dds.sel(pressure=i)
            qv_ds = self.caculate_qv(ds_one)  # qv, qu, qf三个变量同时处理了, 这样做可以吗
            qv = qv_ds.to_array(dim='model')  # 将Dataset 处理为DataArray
            qv_list.append(qv)  # 水汽通量大小, 以及两个矢量

        qv_integral = np.trapz(qv_list[::-1], lev.values[::-1], axis=0)
        da_qv_integral = xr.DataArray(qv_integral, 
                        coords={
                            'model':qv.model.values,
                            'lat':qv.lat.values,
                            'lon':qv.lon.values,
                        }, 
                        dims=['model','lat', 'lon'])
        ds_qv_integral = da_qv_integral.to_dataset(dim='model')
        
        return ds_qv_integral


    def caculate_qv_time(self, ds):
        """计算某一层所有时次的水汽通量
        例如500hPa多个时次的水汽通量
        把不同时次的值聚合在一起而已

        Args:
            ds ([type]): [description]

        Returns:
            [type]: [description]
        """
        pass
        tt = ds.time
        qv_list = []
        for t in tt:
            qvd = self.caculate_qv(ds.sel(time=t))
            qv_list.append(qvd)
        qv_div_dual_time = xr.concat(qv_list, dim='time')
        return qv_div_dual_time
            
    def caculate_qv_integral_time(self, ds):
        """计算多个时次的整层水汽通量散度
        把不同时次的值聚合在一起而已

        Args:
            ds (Dataset): 多时次，多层次的, 含有变量q,v,u

        Returns:
            [type]: [description]
        """
        pass
        tt = ds.time
        qv_integral_list = []
        for t in tt:
            qvdi = self.caculate_qv_integral(ds.sel(time=t))
            qv_integral_list.append(qvdi)
        qv_div_dual_time = xr.concat(qv_integral_list, dim='time')
        return qv_div_dual_time

    def test(self,):
        """用来测试上述函数的准确性的
        """
        pass
        flnm_wrf = '/mnt/zfm_18T/fengxiang/HeNan/Data/high_resolution_high_hgt_upar_d04_latlon.nc'
        ds = xr.open_dataset(flnm_wrf)
        # dds = ds.sel(pressure=500).isel(time=0)
        # dds = ds.sel(pressure=500)
        dds = ds.sel(pressure=[925, 850, 700,500])  # 1000hPa的数据存在很多缺测，故舍弃
        cc = self.caculate_qv_integral_time(ds)
        # da = self.caculate_qv_div_time(dds)
        print(cc.min())

def get_centroid(flnm= '/mnt/zfm_18T/fengxiang/HeNan/Data/1912_900m/rain.nc', t_start='2021-07-20 00', t_end='2021-07-20 12'):
    """获得一段时间降水的质心

    Args:
        flnm (str, optional): 原始wrfout数据的聚合. Defaults to '/mnt/zfm_18T/fengxiang/HeNan/Data/1912_900m/rain.nc'.
        t_start (str, optional): 起始时间. Defaults to '2021-07-20 00'.
        t_end (str, optional): 结束时间. Defaults to '2021-07-20 12'.

    Returns:
        [type]: [description]
    """
    # flnm = '/mnt/zfm_18T/fengxiang/HeNan/Data/1912_900m/rain.nc'
    da = xr.open_dataarray(flnm)
    # tt = slice('2021-07-20 00', '2021-07-20 12')
    tt = slice(t_start, t_end)
    rain = da.sel(time=tt).sum(dim='time')
    lat = sum(sum(rain*rain.lat))/sum(sum(rain))
    lon = sum(sum(rain*rain.lon))/sum(sum(rain))
    lon = lon.round(3)
    lat = lat.round(3)
    return lat, lon

def caculate_average_wrf(da, area = {'lat1':33, 'lat2':34, 'lon1':111.5, 'lon2':113,}):
    """求wrfout数据，在区域area内的区域平均值

    Args:
        da ([type]): 直接利用wrf-python 从wrfout中得到的某个变量的数据

        area = {
            'lat1':33,
            'lat2':34,
            'lon1':111.5,
            'lon2':113,
        }
    Returns:
        [type]: [description]
    """
    lon = da['XLONG'].values
    lat = da['XLAT'].values
    ## 构建掩膜, 范围内的是1， 不在范围的是nan值
    clon = xr.where((lon<area['lon2']) & (lon>area['lon1']), 1, np.nan)
    clat = xr.where((lat<area['lat2']) & (lat>area['lat1']), 1, np.nan)
    da = da*clon*clat
    da_mean = da.mean(dim=['south_north', 'west_east'])
    return da_mean

def uv2wind(u,v):
    """将u,v风转为风向、风速

    Args:
        u ([type]): [description]
        v ([type]): [description]

    Returns:
        [type]: 风速、风向（角度）
    """
    deg = 180.0/np.pi # 角度和弧度之间的转换
    rad = np.pi/180.0
    wind_speed = xr.ufuncs.sqrt(u**2+v**2)
    wind_speed.name = 'wind_speed'
    wind_angle = 180.0+xr.ufuncs.arctan2(u, v)*deg
    wind_angle.name = 'wind_angle'
    # wind_speed
    return wind_speed, wind_angle


def caculate_div(ds):
    """计算单层, 单个时次水汽通量散度

    Args:
        ds (Dataset): 
            lat, lon坐标的多维数组, 包含有q,u,v等变量
            # q: kg/kg
            u: m/s
            v: m/s

    Returns:
        qv_div : 计算好的水汽通量散度
    """
    lon = ds.lon
    lat = ds.lat
    u = ds.u*units('m/s')
    v = ds.v*units('m/s')
    # q = ds.q*10**3*units('g/kg')
    # qv_u = q*u/constants.g
    # qv_v = q*v/constants.g

    dx, dy = ca.lat_lon_grid_deltas(lon.values, lat.values)
    div = ca.divergence(u=u, v=v, dx=dx, dy=dy)
    return div

def concat_dxy(var, index):
    """将二维数据在垂直方向上累加
    变成三维的

    Args:
        var ([type]): 二维变量
        index ([type]): 垂直方向的索引

    Returns:
        var_concat: 累加到一块的数据
    """
    var_list = []
    # index = u.bottom_top.values
    for i in index:
        # print(i)
        aa = var.magnitude  # 转为numpy
        bb = xr.DataArray(aa, dims=['south_north', 'west_east']) # 转为DataArray
        var_list.append(bb)
    var_concat = xr.concat(var_list, dim=pd.Index(index, name='bottom_top'))
    return var_concat

def caculate_div3d(u, v, lon, lat):
    """求wrfout数据中三维的u,v数据对应的散度
    就先原始的wrfout数据吧

    Args:
        u ([type]): 三维
        v ([type]): 三维
        lon ([type]): 二维
        lat ([type]): 二维
        
    Example:
        wrf_file = '/mnt/zfm_18T/fengxiang/HeNan/Data/GWD/d03/gwd0/wrfout_d01_2021-07-19_18:00:00'
        ncfile = Dataset(wrf_file)
        u =  getvar(ncfile, 'ua')
        v =  getvar(ncfile, 'va')
        lon = u.XLONG
        lat = u.XLAT
        div = caculate_div3d(u,v, lon, lat)
        div
    """
    pass
    u = u*units('m/s')
    v = v*units('m/s')
    dx, dy = ca.lat_lon_grid_deltas(lon.values, lat.values)
    ## 重组dx和dy, 其实就是把dx和dy的垂直维度加上，虽然每个垂直层上数据一样, 这是由于metpy计算时的问题导致的
    index = u.bottom_top.values
    ddx = concat_dxy(dx, index)
    ddy = concat_dxy(dy, index)
    dddx = ddx.values*units('m')
    dddy = ddy.values*units('m')
    ### 因为这个函数的问题，所以dx必须是和u维度相对应的
    div = ca.divergence(u=u, v=v, dx=dddx, dy=dddy)
    # div
    div = div.rename('div')
    return div