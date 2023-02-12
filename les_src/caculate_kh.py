#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
计算湍流扩散系数
-----------------------------------------
Time             :2022/05/06 09:48:18
Author           :Forxd
Version          :1.0
'''
# %%
import xarray as xr
import numpy as np
import pandas as pd
import netCDF4 as nc
import wrf
import metpy.calc as cal
from metpy.units import units  # units是units模块中的一个变量名(函数名？类名？)
import matplotlib.pyplot as plt
from baobao.caculate import caculate_q_rh_thetaev
from baobao.timedur import timeit
import calcc   # 空间平滑的Fortran代码

# import cal2 as calcc

# %%

def ftest(da, ftsize):
    ft = np.ones((ftsize,ftsize), dtype=np.float32)
    pad_num = int(ftsize/2)
    daa = np.pad(da, ((pad_num, pad_num),(pad_num, pad_num)), mode="edge")  # 填充边界上的数
    m,n = daa.shape # 获取填充后的输入图像的大小
    # back_da = calcc.caculate.cal_ave2d(daa, ft, m,n,ftsize)  # 传递参数时，数组得在前面
    # back_da = calcc.caculate.cal_ave2d(daa, ft, m,n,ftsize)  # 传递参数时，数组得在前面
    back_da = calcc.cal_area_average.cal_ave2d(daa, ft, m,n,ftsize)  # 传递参数时，数组得在前面
    # print(np.sum(daa-back_da))
    back_da_return = back_da[pad_num:m - pad_num, pad_num:n - pad_num]  # 裁剪
    return back_da_return

@timeit
def cal_average(bb, ftsize):
    ## 创建滤波器
    ftsize = 11
    ft = np.ones((ftsize,ftsize), dtype=np.float32)
    ## 添加/拓展边界上的数
    pad_num = int(11/2)
    bb = np.pad(bb,((0,0), (pad_num, pad_num),(pad_num, pad_num)), mode='edge')
    # print(bb)
    z,y,x = bb.shape
    # back_da = calcc.caculate.cal_ave3d(bb, ft, x,y,z,ftsize)  # 传递参数时，数组得在前面
    back_da = calcc.cal_area_average.cal_ave3d(bb, ft, x,y,z,ftsize)  # 传递参数时，数组得在前面
    ## 筛选原来范围内的数据
    back_da_return = back_da[:,pad_num:y - pad_num, pad_num:x - pad_num]  # 裁剪
    # print(back_da_return)
    return back_da_return

# flnm = '/home/fengxiang/LES/Data/WRF/3dles/zhenzhou/wrfout_d01_2021-07-19_15:00:00'
# %%
# flnm = '/home/fengxiang/LES/Data/WRF/3dles/shenzha/wrfout_d01_2014-08-19_13:00:00'
flnm = '/home/fengxiang/LES/Data/WRF/3dles/shenzha/wrfout_d01_2014-08-20_00:00:00'
ds = xr.open_dataset(flnm)
wrfnc = nc.Dataset(flnm)
## 获取变量
zs = wrf.getvar(wrfnc, 'zstag')
z = wrf.getvar(wrfnc, 'z')
u = wrf.getvar(wrfnc, 'ua')
v = wrf.getvar(wrfnc, 'va')
w = ds['W'].squeeze()
w = w[1:-1,:,:]   # 通量层，99层
temp = wrf.getvar(wrfnc, 'temp', units='degC')
td = wrf.getvar(wrfnc, 'td')
rh = wrf.getvar(wrfnc, 'rh')
pres = wrf.getvar(wrfnc, 'pressure')
pres = units.Quantity(pres.values, "hPa")
td = units.Quantity(td.values, 'degC')
q = cal.specific_humidity_from_dewpoint(pres, td).magnitude*10**3  #(g/kg)
# q = cal.specific_humidity_from_dewpoint(pres, td)  #(g/kg)
mx = cal.mixing_ratio_from_relative_humidity(pres,temp,rh)
theta_v = cal.virtual_potential_temperature(pres, temp,mx)
# %%
# type(q)

# %%
## 计算平均值和扰动量
def caculate_bar_derivative(da):
    da_bar = cal_average(da.astype('float32'), 11)
    # da_bar = xr.DataArray(da_bar, coords=da.coords, dims=da.dims)
    da_der = da-da_bar # w derivative w的导数
    return da_bar, da_der  # 平均值， 扰动值

ub, ud = caculate_bar_derivative(u.values)
vb, vd = caculate_bar_derivative(v.values)
wb, wd = caculate_bar_derivative(w.values)
qb, qd = caculate_bar_derivative(q)
thetavb, thetavd = caculate_bar_derivative(theta_v.values)
# %%
## 将q', u', theta', v'插值到通量层上
# print(ub)
# print(vd)

# %%
index_k = list(u.bottom_top.values[0:-1])
wud = np.ones(wb.shape)  # w'u', 这里的wb是把最底层和最高层去了之后的
wqd = np.ones(wb.shape)
wthetavd = np.ones(wb.shape)

duz = np.ones(wb.shape).astype('float64')
dqz = np.ones(wb.shape).astype('float64')
dthetavz = np.ones(wb.shape).astype('float64')

# %%
z
# duz
# q = q.astype('float32')

# %%
# @timeit
# def ttt():
for k in index_k:   # 0,1,2, ...98
    # print(k)
    ## 将通量ub数据插值到通量层上
    wud[k,:,:] = wd[k,:,:]*0.5*(ud[k+1,:,:]+ud[k,:,:])  # 动量通量
    wqd[k,:,:] = wd[k,:,:]*0.5*(qd[k+1,:,:]+qd[k,:,:])  # 水汽通量
    # print(wd.shape)
    # print(wqd.shape)
    wthetavd[k,:,:] = wd[k,:,:]*0.5*(thetavd [k+1,:,:]+thetavd[k,:,:])  # 热量通量

    # dz = z[k+1,:,:]-z[k,:,:]
    # du = u[k+1,:,:]-u[k,:,:]
    print(abs(q[k+1,:,:]-q[k,:,:]).min())
    duz[k,:,:] = (ub[k+1,:,:]-ub[k,:,:])/(z.values[k+1,:,:]-z.values[k,:,:])
    dqz[k,:,:] = (qb[k+1,:,:]-qb[k,:,:])/(z.values[k+1,:,:]-z.values[k,:,:])
    dthetavz[k,:,:] = (thetavb[k+1,:,:]-thetavb[k,:,:])/(z.values[k+1,:,:]-z.values[k,:,:])

    

# wud.shape
# %%
n,m,l = dqz.shape  # 分别对应z,y,x, 这里的z和高度的变量冲突了
km = calcc.cal_area_average.cal_coefficient(wud,duz,l,m,n)
kq = calcc.cal_area_average.cal_coefficient(wqd,dqz,l,m,n)
kh = calcc.cal_area_average.cal_coefficient(wthetavd,dthetavz,l,m,n)
# duz.shape
# %%
# wthetavd
# kh.max()
kh = xr.DataArray(kh, coords=w.coords, dims=w.dims)
kq = xr.DataArray(kq, coords=w.coords, dims=w.dims)
km = xr.DataArray(km, coords=w.coords, dims=w.dims)
# kh
# %%
# flag = 9999999999
flag = 1000
khna = xr.where((kh<flag)&(kh>(-flag)), kh,np.nan)
kqna = xr.where((kq<flag)&(kq>(-flag)), kq,np.nan)
kmna = xr.where((km<flag)&(km>(-flag)), km,np.nan)

# khna = xr.where(kh<9999999999, kh,np.nan)
# kqna = xr.where((kq<1)&(kq>(-1)), kh,np.nan)
# kmna = xr.where((km<1)&(km>(-1)), kh,np.nan)
# khna.min()
# %%

kh2 = khna.interpolate_na(dim='bottom_top_stag', method='linear')
kq2 = kqna.interpolate_na(dim='bottom_top_stag', method='linear')
km2 = kmna.interpolate_na(dim='bottom_top_stag', method='linear')
kh2
# kh2
# %%
kh.min()
# kh2.mean(dim=['south_north', 'west_east']).plot()
# kq2.mean(dim=['south_north', 'west_east']).plot()
# kq2
kq.mean(dim=['south_north', 'west_east']).plot()
# %%
kq.max()
# %%
# wd


wuda = xr.DataArray(wud, coords=w.coords, dims=w.dims)
wqda = xr.DataArray(wqd, coords=w.coords, dims=w.dims)
wthetavda = xr.DataArray(wthetavd, coords=w.coords, dims=w.dims)
wthetavda.mean(dim=['south_north', 'west_east']).plot()
# %%
# km.mean(dim=['south_north', 'west_east']).plot()
# kh.mean(axis=1).mean(axis=1).plot()
# km.mean(axis=1).mean(axis=1).plot()
# kq.mean(axis=1).mean(axis=1).plot()
# kqna-kmna
# q_mean
# %%

theta_v_mean = theta_v.mean(dim=['south_north', 'west_east'])
# u_mean = u.mean(dim=['south_north', 'west_east'])
u_mean=ub.mean(axis=1).mean(axis=1)
q_mean=q.mean(axis=1).mean(axis=1)
# q_mean = theta_v.mean(dim=['south_north', 'west_east'])
height = z.mean(dim=['south_north', 'west_east']).values
height_s = zs.mean(dim=['south_north', 'west_east']).values[1:-1]

# %%
wud_mean = wud.mean(axis=1).mean(axis=1)
wqd_mean = wqd.mean(axis=1).mean(axis=1)
wthetavd_mean = wthetavd.mean(axis=1).mean(axis=1)
wud_mean.shape
# %%
cm = 1/2.54
fig = plt.figure(figsize=[17*cm, 8*cm], dpi=600)
ax1 = fig.add_axes([0.1, 0.2, 0.2, 0.75])
ax2 = fig.add_axes([0.38, 0.2, 0.2, 0.75])
ax3 = fig.add_axes([0.65, 0.2, 0.2, 0.75])
ax1.plot(u_mean,height, color='black')
ax1.set_xticks([0,1,2])
ax2.plot(theta_v_mean,height,color='red')
ax3.plot(q_mean,height, color='blue')
# ax2.
ax1.set_ylim(0,2000)
ax2.set_ylim(0,2000)
ax3.set_ylim(0,2000)

ax2.set_xlabel(r"$\theta_v \quad (K)$", )
ax1.set_xlabel(r"$u \quad (m/s)$")
ax3.set_xlabel(r"$q \quad (g/kg)$")
ax1.set_ylabel('Height (m)')
fig.savefig('mean.png')
# %%
cm = 1/2.54
fig = plt.figure(figsize=[8*cm, 8*cm], dpi=600)
ax = fig.add_axes([0.2, 0.15, 0.8, 0.7])
ax.plot(wud_mean,height_s, label=r"$\overline{w'u'}$", color='black')
ax.plot(wqd_mean,height_s, label=r"$\overline{w'q'}$", color='blue')
ax.plot(wthetavd_mean,height_s, label=r"$\overline{w'\theta'}$", color='red')
ax.set_ylabel('Height (m)')
ax.legend(edgecolor='white')
ax.set_yticks(np.arange(0,2000+1,250))
ax.set_ylim(0,2000)
fig.savefig('flux.png')
# %%
khna.max()
# %%
kh_mean = -1*khna.mean(dim=['south_north', 'west_east']).values
kq_mean = -1*kqna.mean(dim=['south_north', 'west_east']).values
km_mean = -1*kmna.mean(dim=['south_north', 'west_east']).values
# kh_mean = khna.mean(dim=['south_north', 'west_east']).values
kh_mean
cm = 1/2.54
fig = plt.figure(figsize=[8*cm, 8*cm], dpi=600)
ax = fig.add_axes([0.2, 0.2, 0.75, 0.68])
ax.plot(km_mean,height_s, label=r"$K_m$", color='black')
ax.plot(kh_mean,height_s, label=r"$K_h$", color='red')
ax.plot(kq_mean,height_s, label=r"$K_q$", color='blue')
ax.legend(edgecolor='white')
# ax.set_ylim(0,600)
# ax.set_yticks(np.arange(0,1000+1,100))
# ax.set_ylim(0,1000)
ax.set_yticks(np.arange(0,2000+1,250))
ax.set_ylim(0,2000)
ax.set_xticks(np.arange(-20, 150, 20))
ax.set_ylabel('Height (m)')
fig.savefig('kh2')

# ax.plot(wqd_mean,height_s, label=r"$\overline{w'q'}$")
# ax.plot(wthetavd_mean,height_s, label=r"$\overline{w'\theta'}$")
# ax.set_ylabel('Height (m)')
# ax.legend(edgecolor='white')
# ax.set_ylim(0,2000)
# wthetavd_mean.min()
# t
# ax.plot(bb*(-1),yy)
# ax.plot(cc*(-1),yy)


# khna

# km
# kq
# kh
# co.max()
# coo = np.where(co<1, co,np.nan)
# coo.max()
# coo.fillna(method='ffill')

# %%
# dqz
# u.bottom_top
# kh.max()
# abs(dqz).min()
# abs(dthetavz).min()
# abs(duz).min()
# z.values[k+1,:,:]-z.values[k,:,:]
# np.min(abs(q[k+1,:,:]-q[k,:,:])/(z.values[k+1,:,:]-z.values[k,:,:]))
# wqd
# wud
# duz.min()
# np.where(duz==0)
# wqd
# wd.shape
# qd.shape
# duz
# z.values
# dqz.shape
# wqd.shape
# wud.shape
# qd.shape


    ## 梯度










# height_stag
# height_stag.renaame
# b = z.transpose(2, 1, 0)
# c = zs.transpose(2, 1, 0)
# a = a.transpose(2, 1, 0)
# wrf.interpz3d(ub, z, zs[1:-1, :, :])
# wrf.interpz3d(ub, z.values, height_stag)
# ub.shape.transpose()
# a.values
# b.values
# a
# ub.shape
# z.shape
# zs[1:-1, :, :]
# z
# zs


#%%
# ub


# %%
### 求通量项
##w'
# w = ds['W'].squeeze()

'''
dd = cal_average(w.values.astype('float32'), 11)
wbar = xr.DataArray(dd, coords=w.coords, dims=w.dims)
wbar.max()
wd = w-wbar # w derivative w的导数
wd = wd[1:-1,:,:]   # 通量层，99层
wd

# %%
## u',v', q', theta'
u = wrf.getvar(wrfnc, 'ua')
v = wrf.getvar(wrfnc, 'va')
# q = wrf.getvar(wrfnc, 'q')
theta = wrf.getvar(wrfnc, 'theta')  # 质量层上的垂直速度
theta
ubar = cal_average(u.values.astype('float32'), 11)
vbar = cal_average(v.values.astype('float32'), 11)
# qbar = cal_average(q.values.astype('float32'), 11)
thetabar = cal_average(w.values.astype('float32'), 11)

# %%



u = ds['V'].squeeze()
dd = cal_average(u.values.astype('float32'), 11)
ubar = xr.DataArray(dd, coords=u.coords, dims=u.dims)
ubar.max()
# %%

# pad_num = int(11/2)
# bb.shape
# cc = np.pad(bb,((0,0), (pad_num, pad_num),(pad_num, pad_num)), mode='edge')
# cc
# %%
# q.units
# theta_v
# mx
# theta_v
# q
q


# q
# q.min()
'''

# %%
'''


# %%


# %%
# ds['P'].values
# %%
# ds['QFX']
# cal.kinematic_flux()
wrfnc = nc.Dataset(flnm)
zs = wrf.getvar(wrfnc, 'zstag')
z = wrf.getvar(wrfnc, 'z')
u = wrf.getvar(wrfnc, 'ua')
v = wrf.getvar(wrfnc, 'va')
w = wrf.getvar(wrfnc, 'wa')
q = wrf.getvar(wrfnc, 'pressure')
# q
w
# %%
q
# %%
q.values.min()
# %%

# pre_level = [900, 925, 850, 700, 600, 500, 200]
pre_level = np.arange(950, 790, -10)
dds = xr.Dataset()
data_nc = wrfnc
p = wrf.getvar(data_nc, 'pressure', squeeze=False)

for var in ['ua', 'va', 'wa','td', 'temp', 'height_agl', 'z','geopt']:
    if var in ['temp', 'td', 'theta', 'theta_e', ]:
        da = wrf.getvar(data_nc, var, squeeze=False, units='degC')
    else:
        da = wrf.getvar(data_nc, var, squeeze=False)
    # dds[var] = da.expand_dims(dim='Time')
    dds[var] = wrf.interplevel(da, p, pre_level, squeeze=False,)
    # dds[var] = da
dds


# %%
ds_upar = dds.rename({'level':'pressure', 'XLAT':'lat', 'XLONG':'lon', 'Time':'time'})
# ds_upar = dds_concate.rename({'level':'pressure', 'Time':'time'})
ds_upar = ds_upar.drop_vars(['XTIME'])
cc = caculate_q_rh_thetaev(ds_upar)
# ds_r
# ds_r = xr.merge([ds_upar, cc])
# cc
dss = xr.merge([ds_upar, cc])
# %%
dss['theta_v'].squeeze()

# %%
## TODO 需要的是   
# dss['theta_v'].isel(south_north=10, west_east=10)
# ds_upar

# %%
dss = dss.squeeze()
bt = np.arange(len(dss.pressure))
dss1 = dss.assign_coords({'bottom_top':('pressure',bt)})
dss2 = dss1.swap_dims({'pressure':'bottom_top'})
dss = dss2
# z.isel(south_north=10, west_east=10)
# %%
u = dss['ua']
v = dss['va']
w = dss['wa']
th = dss['theta_v']
q = dss['q']
z = dss['z']


## 计算区域平均值
um = dss['ua'].mean(dim=['south_north','west_east'])
vm = dss['va'].mean(dim=['south_north','west_east'])
wm = dss['wa'].mean(dim=['south_north','west_east'])
thm = dss['theta_v'].mean(dim=['south_north','west_east'])
qm = dss['q'].mean(dim=['south_north','west_east'])
# thm 

# %%
dss['theta_v'].max()
# %%
## 计算上下层之间的差值
uu = dss['ua'].copy()
vv = dss['va'].copy()
ww = dss['wa'].copy()
thh = dss['theta_v'].copy()
qq = dss['q'].copy()
# len(um)
# %%
qq

# %%
thh.max()
# %%

## u-ubar
for i in range(len(um)):
    uu[i,:,:] = (u.isel(bottom_top=i)-um[i]).values
    vv[i,:,:] = (v.isel(bottom_top=i)-vm[i]).values
    ww[i,:,:] = (w.isel(bottom_top=i)-wm[i]).values
    thh[i,:,:] = (th.isel(bottom_top=i)-thm[i]).values
    qq[i,:,:] = (q.isel(bottom_top=i)-qm[i]).values

# %%
thh.max()
# %%
## 计算通量和K值

# u = dss['ua']
du = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])  # 上下层之间的速度差
dv = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])  # 上下层之间的速度差
dw = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])  # 
dth = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])  # 
dq = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])  # 
dz = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])  # 上下层之间的高度

duz = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])  # du/dz
dvz = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])   # dv/dz
dthz = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])   # dv/dz
dqz = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])   # dv/dz
# dwz = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])
ku = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])  # 通量层上的扩散系数
kh = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])  # 通量层上的扩散系数
kq = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])  # 通量层上的扩散系数

duu = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])  # 通量层上的，通量
dthh = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])  # 通量层上的，通量
dqq = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])  # 通量层上的，通量
# dw
# %%
for i in range(len(um)-1):
    # print(du[i])
    du[i,:,:] = u[i+1,:,:]-u[i,:,:]
    dth[i,:,:] = th[i+1,:,:]-th[i,:,:]
    dq[i,:,:] = q[i+1,:,:]-q[i,:,:]

    dz[i,:,:] = z[i+1,:,:]-z[i,:,:]

    duz[i,:,:] = du[i,:,:]/dz[i,:,:]
    dthz[i,:,:] = dth[i,:,:]/dz[i,:,:]
    dqz[i,:,:] = dq[i,:,:]/dz[i,:,:]

    duu[i,:,:] = 0.5*(uu[i+1,:,:]+uu[i,:,:])
    dthh[i,:,:] = 0.5*(thh[i+1,:,:]+thh[i,:,:])
    dqq[i,:,:] = 0.5*(qq[i+1,:,:]+qq[i,:,:])

    

    ku[i,:,:] = duu[i,:,:]/duz[i,:,:]
    kh[i,:,:] = dthh[i,:,:]/dthz[i,:,:]
    kq[i,:,:] = dqq[i,:,:]/dqz[i,:,:]



# %%
np.nanmax(thh)
# dthh
# %%

aa = ku.mean(axis=1).mean(axis=1)
bb = kh.mean(axis=1).mean(axis=1)
cc = kq.mean(axis=1).mean(axis=1)
yy = np.arange(len(aa)+1, 1, -1)
cm = 1/2.54
fig = plt.figure(figsize=[8*cm, 8*cm], dpi=300)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(aa*(-1)*(-1),yy)
ax.plot(bb*(-1),yy)
ax.plot(cc*(-1),yy)

# %%
# np.nanmin(aa)
# dss['z']
bb
# duz

# cc
# ku












# %%
# w
# u = 
# ds['z']
# np.mean(u)
um = u.mean(dim=['south_north','west_east'])
vm = v.mean(dim=['south_north','west_east'])
wm = w.mean(dim=['south_north','west_east'])

# %%
# u.bottom_top
uu = u.copy()
vv = v.copy()
ww = w.copy()

# len(um)
for i in range(len(um)):
    # print(i)
    # print(um[i])
    # uu.isel(bottom_top=i).values = (u.isel(bottom_top=i)-um[i]).values
    uu[i,:,:] = (u.isel(bottom_top=i)-um[i]).values
    
    vv[i,:,:] = (v.isel(bottom_top=i)-vm[i]).values
    ww[i,:,:] = (w.isel(bottom_top=i)-wm[i]).values

# %%
# uu
# du 

# u.shape[0]
du = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])  # 上下层之间的速度差
dv = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])  # 上下层之间的速度差
dw = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])  # 
dz = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])  # 上下层之间的高度
duz = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])  # du/dz
dvz = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])   # dv/dz
# dwz = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])


ku = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])  # 通量层上的扩散系数
duu = np.empty([u.shape[0]-1, u.shape[1], u.shape[2]])  # 通量层上的，通量
# dw
# %%
for i in range(len(um)-1):
    # print(du[i])
    du[i,:,:] = u[i+1,:,:]-u[i,:,:]
    dz[i,:,:] = z[i+1,:,:]-z[i,:,:]
    duz[i,:,:] = du[i,:,:]/dz[i,:,:]
    duu[i,:,:] = 0.5*(uu[i+1,:,:]+uu[i,:,:])

    ku[i,:,:] = duu[i,:,:]/duz[i,:,:]

# %%
# duz
# duz.min()
aa = ku.mean(axis=1).mean(axis=1)
yy = np.arange(len(aa)+1, 1, -1)
# %%
# len(yy)
yy
# %%
# (duu[1,:,:]/duz[1,:,:]).shape
import matplotlib.pyplot as plt
cm = 1/2.54
fig = plt.figure(figsize=[8*cm, 8*cm], dpi=300)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(aa*10**4*(-1),yy)
# ax.invert_yaxis()
    


    
    

# w.mean()-ww.mean()
# u.mean()-uu.mean()
# v.mean()-vv.mean()
# w
# uu.mean()

# uu[0,0,0] = 1
# uu[0,0,0]
# u
# uu[0,:,:]
# uu







# z



# %%
'''