import pandas as pd
import os
import numpy as np

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

def select_cmap(flag):
    """
    flag : 选择哪个色标    

    """
    path = os.path.dirname(os.path.abspath(__file__))
    if flag == 'rain9':
        flnm = path+'/colortxt/9colors_rain.rgb'
        rgb = get_rgb(flnm)
        
    return rgb
