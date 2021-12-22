#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
根据第三次科考的探空观测资料,
将探空资料转为micaps站点的格式



55591   91.13   29.67 3650   90
    气压     高度      温度    露点   风向    风速
     654    9999      19       7     270       3
     512    9999       2      -1    9999    9999
     500     588       1      -3     215       4
     400     764     -10     -13     290       3
     363    9999     -15     -16    9999    9999
     300     980     -24     -29     340       6
     266    9999     -31     -48    9999    9999
     250    1111     -34     -53      15       3
     221    9999     -41     -60    9999    9999
     200    1263     -47     -65      55       6
     152    9999     -62    9999    9999    9999
     150    1447     -63    9999      60       7
     104    9999     -79    9999    9999    9999
     101    9999     -78    9999    9999    9999
     100    1686     -78    9999      40      10

-----------------------------------------
Time             :2021/03/22 21:25:58
Author           :Forxd
Version          :1.0
'''

# %%
import numpy as np
import pandas as pd
import xarray as xr
import pandas as pd




# %%
def read_station(flnm):
    """读取站点数据，返回各物理量
    读的是第三次科考的数据

    Args:
        flnm ([type]): 站点文件名

    Returns:
        [type]: 各物理量
    """
    col_names = ['temp', 'height', 'pressure', 'td', 'wind_angle', 'wind_speed']
    # df = pd.read_fwf(flnm,
    #                  skiprows=39,
    #                  usecols=[2, 6, 7, 8, 10, 11],
    #                  names=col_names)
    df = pd.read_csv(flnm,
                     sep='\t',
                     skiprows=39,
                     usecols=[2, 6, 7, 8, 10, 11],
                     names=col_names)
    df[df <= -32678] = None
    # 将含有缺省值的行略去，how=any即只要该行含有Nan,略去整行

    
    ## 只取上升段的数据
    index = df['height'].idxmax()  # 最大值所在的行
    df = df[0:index]

    ## 去除掉缺省值
    df = df.dropna(axis=0,
                #    subset=('T', 'v', 'u', 'Height', 'P', 'TD'),
                   subset=('temp', 'height', 'pressure', 'td', 'wind_angle', 'wind_speed'),
                   how='any').reset_index(drop=True)




    ## 改变各列的顺序
    order = ['pressure', 'height', 'temp', 'td', 'wind_angle', 'wind_speed']
    df = df[order]



    ## 获取经纬度，以初始时刻的经纬度为所有时次的经纬度
    a = pd.read_csv(flnm, sep='\t', skiprows=39)
    b = a.values[0]
    lat = b[-5]
    lon = b[-6]

    
    
    ## 改变数据的单位
    df['td'] = (df['td']-273.15).round(1)   
    df['temp'] = (df['temp']-273.15).round(1)   
    df['height'] = (df['height']/10).round(1)
    

    return df , lon, lat



flnm = '/mnt/zfm_18T/fengxiang/DATA/UPAR/upar_20140819/GPS_55472_20140819-0715.tsv'
df1,lon, lat = read_station(flnm)
file_save = '/mnt/zfm_18T/fengxiang/DATA/UPAR/20140819/shenzha_2014-08-19_00.txt'
df1.to_csv(file_save, sep='\t', index=False, header=False,line_terminator='\n',encoding='utf-8')

## 添加首行内容
first_line = '55472   {}   {}  \n'.format(lon, lat)
with open(file_save, 'r+') as f:
    content = f.read()
    # content=content.encode("utf-8")
    f.seek(0,0)
    # f.write('writer:Fatsheep\n'+content)
    f.write(first_line+content)
    # f.wrtite('zhe ge ')


# %%
# aa
# ss = df1['height'][aa]
# ss
# aa
# df2 = df1[0:aa]
# df2
# df1['height'][1500]
# for i in ss:
    # print(i)
# ss.max()    
# df1['height'][aa]