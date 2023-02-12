#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
根据加密探空的tsv文件，得到WRF-LES需要的探空格式数据， input-sounding


序号  参数      Record name:    Unit:           Data type:          Divisor: Offset:
---------------------------------------------------------------------------------
0    时间         time            sec             float (4)          1        0       
1    对数压力     Pscl            ln              short (2)          1        0       
2    温度         T               K               short (2)          10       0       
3    相对湿度     RH              %               short (2)          1        0       
4    V向风        v               m/s             short (2)          -100     0       
5    U向风        u               m/s             short (2)          -100     0       
6    海拔高度     Height          m               short (2)          1        30000   
7    气压         P               hPa             short (2)          10       0       
8    露点温度     TD              K               short (2)          10       0       
9    比湿         MR              g/kg            short (2)          100      0       
10   风向         DD              dgr             short (2)          1        0       
11   风速         FF              m/s             short (2)          10       0       
12   水平角       AZ              dgr             short (2)          1        0       
13   距离         Range           m               short (2)          0.01     0       
14   经度         Lon             dgr             short (2)          100      0       
15   纬度         Lat             dgr             short (2)          100      0       
16   预留字段     SpuKey          bitfield        unsigned short (2) 1        0       
17   预留字段     UsrKey          bitfield        unsigned short (2) 1        0       
18   雷达高度     RadarH          m               short (2)          1        30000   
-----------------------------------------
Time             :2022/05/11 16:24:19
Author           :Forxd
Version          :1.0
'''
# %%
from pkgutil import get_data
import numpy as np
import xarray as xr
import pandas as pd
import metpy
from metpy.units import units
from metpy.calc import potential_temperature
# %%
def get_sounding(flnm):
    df = pd.read_csv(flnm)
    col_names = ['temp', 'v','u', 'height', 'pressure', 'q']
    df = pd.read_csv(flnm,
                        sep='\t',
                        skiprows=39,
                        usecols=[2,4, 5, 6, 7, 9,],
                        names=col_names)
    df[df <= -32678] = None
    # 将含有缺省值的行略去，how=any即只要该行含有Nan,略去整行

    ## 只取上升段的数据
    index = df['height'].idxmax()  # 最大值所在的行
    df = df[0:index]

    ## 去除掉缺省值
    df = df.dropna(axis=0,
                #    subset=('T', 'v', 'u', 'Height', 'P', 'TD'),
                    subset=('temp', 'height', 'pressure', 'q', 'u', 'v'),
                    how='any').reset_index(drop=True)

    ## 计算位温
    pressure = units.Quantity(df['pressure'].values, "hPa")
    temp = units.Quantity(df['temp'].values, "K")
    df['theta'] = potential_temperature(pressure, temp)

    ##  只要多少高度之内的数据
    dff = df[df['height']<df['height'][1]+2500]
    return dff

def write_les(df, flsave):
# df
    ## 获取需要的数据，并重新排序
    ddf = df.iloc[1:] # 这个是高空的数据
    order = ['height', 'theta', 'q', 'u', 'v']
    ddf = ddf[order].round(2)

    head = df.iloc[0]   # 这个是地面的数据
    str_head = '{} {} {} \n'.format(head['pressure'].round(2),head['theta'].round(2), head['q'].round(2), )

    ddf['height'] = ddf['height']-head['height']

    ## 将首行数据加上
    ddf.to_csv(flsave, header=False, index=False, sep=' ', line_terminator='\n', encoding='utf-8')
    with open(flsave, 'r+') as f:
        content = f.read()
        # content=content.encode("utf-8")
        f.seek(0,0)
        # f.write('writer:Fatsheep\n'+content)
        f.write(str_head+content)
        # f.wrtite('zhe ge ')

flnm = '/home/fengxiang/LES/Data/OBS/sounding_shenzha/GPS_55472_20140819-1915.tsv'
flsave = '/home/fengxiang/LES/Data/OBS/sounding_shenzha/input_sounding_shenzha_20140819-1915'
df = get_sounding(flnm)
write_les(df, flsave)

# %%
df

# %%
import matplotlib.pyplot as plt
cm = 1/2.54
fig = plt.figure(figsize=[8*cm, 8*cm], dpi=300)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(df['theta'].values[1:], df['height'][1:])
# %%
# dff['theta']
# dff['theta'].plot()
ax.plot([1,2,3], [4,5,6])
# plt.plot()
# dff['height']






