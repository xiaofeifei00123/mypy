
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
