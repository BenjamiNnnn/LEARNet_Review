import tensorflow as tf

# 491
def test_0():
    img_lis = []
    for i in range(1, 492):
        img_path_cn = './test/0/{}_0.jpg'.format(i)
        img_raw = tf.io.read_file(img_path_cn)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        # 根据模型调整其大小
        img_final = tf.image.resize(img_tensor, [112, 112])
        img_final = img_final / 112.0
        img_lis.append(img_final)
    return img_lis

# 55
def test_1():
    img_lis = []
    for i in range(1, 56):
        img_path_cn = './test/1/{}_1.jpg'.format(i)
        img_raw = tf.io.read_file(img_path_cn)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        # 根据模型调整其大小
        img_final = tf.image.resize(img_tensor, [112, 112])
        img_final = img_final / 112.0
        img_lis.append(img_final)
    return img_lis

# 528
def test_2():
    img_lis = []
    for i in range(1, 529):
        img_path_cn = './test/2/{}_2.jpg'.format(i)
        img_raw = tf.io.read_file(img_path_cn)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        # 根据模型调整其大小
        img_final = tf.image.resize(img_tensor, [112, 112])
        img_final = img_final / 112.0
        img_lis.append(img_final)
    return img_lis

# 879
def test_3():
    img_lis = []
    for i in range(1, 880):
        img_path_cn = './test/3/{}_3.jpg'.format(i)
        img_raw = tf.io.read_file(img_path_cn)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        # 根据模型调整其大小
        img_final = tf.image.resize(img_tensor, [112, 112])
        img_final = img_final / 112.0
        img_lis.append(img_final)
    return img_lis

# 594
def test_4():
    img_lis = []
    for i in range(1, 595):
        img_path_cn = './test/4/{}_4.jpg'.format(i)
        img_raw = tf.io.read_file(img_path_cn)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        # 根据模型调整其大小
        img_final = tf.image.resize(img_tensor, [112, 112])
        img_final = img_final / 112.0
        img_lis.append(img_final)
    return img_lis

# 416
def test_5():
    img_lis = []
    for i in range(1, 417):
        img_path_cn = './test/5/{}_5.jpg'.format(i)
        img_raw = tf.io.read_file(img_path_cn)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        # 根据模型调整其大小
        img_final = tf.image.resize(img_tensor, [112, 112])
        img_final = img_final / 112.0
        img_lis.append(img_final)
    return img_lis

# 626
def test_6():
    img_lis = []
    for i in range(1, 627):
        img_path_cn = './test/6/{}_6.jpg'.format(i)
        img_raw = tf.io.read_file(img_path_cn)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        # 根据模型调整其大小
        img_final = tf.image.resize(img_tensor, [112, 112])
        img_final = img_final / 112.0
        img_lis.append(img_final)
    return img_lis
#-------------------------------------------------------------------------------
# 467
def valid_0():
    img_lis = []
    for i in range(1, 468):
        img_path_cn = './val/0/{}_0.jpg'.format(i)
        img_raw = tf.io.read_file(img_path_cn)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        # 根据模型调整其大小
        img_final = tf.image.resize(img_tensor, [112, 112])
        img_final = img_final / 112.0
        img_lis.append(img_final)
    return img_lis

# 56
def valid_1():
    img_lis = []
    for i in range(1, 57):
        img_path_cn = './val/1/{}_1.jpg'.format(i)
        img_raw = tf.io.read_file(img_path_cn)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        # 根据模型调整其大小
        img_final = tf.image.resize(img_tensor, [112, 112])
        img_final = img_final / 112.0
        img_lis.append(img_final)
    return img_lis

# 496
def valid_2():
    img_lis = []
    for i in range(1, 497):
        img_path_cn = './val/2/{}_2.jpg'.format(i)
        img_raw = tf.io.read_file(img_path_cn)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        # 根据模型调整其大小
        img_final = tf.image.resize(img_tensor, [112, 112])
        img_final = img_final / 112.0
        img_lis.append(img_final)
    return img_lis

# 895
def valid_3():
    img_lis = []
    for i in range(1, 896):
        img_path_cn = './val/3/{}_3.jpg'.format(i)
        img_raw = tf.io.read_file(img_path_cn)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        # 根据模型调整其大小
        img_final = tf.image.resize(img_tensor, [112, 112])
        img_final = img_final / 112.0
        img_lis.append(img_final)
    return img_lis

# 653
def valid_4():
    img_lis = []
    for i in range(1, 654):
        img_path_cn = './val/4/{}_4.jpg'.format(i)
        img_raw = tf.io.read_file(img_path_cn)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        # 根据模型调整其大小
        img_final = tf.image.resize(img_tensor, [112, 112])
        img_final = img_final / 112.0
        img_lis.append(img_final)
    return img_lis

# 415
def valid_5():
    img_lis = []
    for i in range(1, 416):
        img_path_cn = './val/5/{}_5.jpg'.format(i)
        img_raw = tf.io.read_file(img_path_cn)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        # 根据模型调整其大小
        img_final = tf.image.resize(img_tensor, [112, 112])
        img_final = img_final / 112.0
        img_lis.append(img_final)
    return img_lis

# 607
def valid_6():
    img_lis = []
    for i in range(1, 608):
        img_path_cn = './val/6/{}_6.jpg'.format(i)
        img_raw = tf.io.read_file(img_path_cn)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        # 根据模型调整其大小
        img_final = tf.image.resize(img_tensor, [112, 112])
        img_final = img_final / 112.0
        img_lis.append(img_final)
    return img_lis

#-------------------------------------------------------------------------------
# 3995
def train_0():
    img_lis = []
    for i in range(1, 3996):
        img_path_cn = './train/0/{}_0.jpg'.format(i)
        img_raw = tf.io.read_file(img_path_cn)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        # 根据模型调整其大小
        img_final = tf.image.resize(img_tensor, [112, 112])
        img_final = img_final / 112.0
        img_lis.append(img_final)
    return img_lis

# 436
def train_1():
    img_lis = []
    for i in range(1, 437):
        img_path_cn = './train/1/{}_1.jpg'.format(i)
        img_raw = tf.io.read_file(img_path_cn)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        # 根据模型调整其大小
        img_final = tf.image.resize(img_tensor, [112, 112])
        img_final = img_final / 112.0
        img_lis.append(img_final)
    return img_lis

# 4097
def train_2():
    img_lis = []
    for i in range(1, 4098):
        img_path_cn = './train/2/{}_2.jpg'.format(i)
        img_raw = tf.io.read_file(img_path_cn)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        # 根据模型调整其大小
        img_final = tf.image.resize(img_tensor, [112, 112])
        img_final = img_final / 112.0
        img_lis.append(img_final)
    return img_lis

# 7215
def train_3():
    img_lis = []
    for i in range(1, 7216):
        img_path_cn = './train/3/{}_3.jpg'.format(i)
        img_raw = tf.io.read_file(img_path_cn)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        # 根据模型调整其大小
        img_final = tf.image.resize(img_tensor, [112, 112])
        img_final = img_final / 112.0
        img_lis.append(img_final)
    return img_lis

# 4830
def train_4():
    img_lis = []
    for i in range(1, 4831):
        img_path_cn = './train/4/{}_4.jpg'.format(i)
        img_raw = tf.io.read_file(img_path_cn)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        # 根据模型调整其大小
        img_final = tf.image.resize(img_tensor, [112, 112])
        img_final = img_final / 112.0
        img_lis.append(img_final)
    return img_lis

# 3171
def train_5():
    img_lis = []
    for i in range(1, 3172):
        img_path_cn = './train/5/{}_5.jpg'.format(i)
        img_raw = tf.io.read_file(img_path_cn)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        # 根据模型调整其大小
        img_final = tf.image.resize(img_tensor, [112, 112])
        img_final = img_final / 112.0
        img_lis.append(img_final)
    return img_lis

# 4965
def train_6():
    img_lis = []
    for i in range(1, 4966):
        img_path_cn = './train/6/{}_6.jpg'.format(i)
        img_raw = tf.io.read_file(img_path_cn)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
        # 根据模型调整其大小
        img_final = tf.image.resize(img_tensor, [112, 112])
        img_final = img_final / 112.0
        img_lis.append(img_final)
    return img_lis

# 加载数据集
def load_test_data():
    imglis_0 = test_0()  # (491,112,112,3)
    imglis_1 = test_1()  # (55,112,112,3)
    imglis_2 = test_2()  # (528,112,112,3)
    imglis_3 = test_3()  # (879,112,112,3)
    imglis_4 = test_4()  # (594,112,112,3)
    imglis_5 = test_5()  # (416,112,112,3)
    imglis_6 = test_6()  # (626,112,112,3)

    img0_stack = tf.stack([imglis_0[i] for i in range(491)])
#     print(img0_stack.shape)  # (491,112,112,3)
    img1_stack = tf.stack([imglis_1[i] for i in range(55)])
#     print(img1_stack.shape)  # (55,112,112,3)
    img2_stack = tf.stack([imglis_2[i] for i in range(528)])
#     print(img2_stack.shape)  # (528,112,112,3)
    img3_stack = tf.stack([imglis_3[i] for i in range(879)])
#     print(img3_stack.shape)  # (879,112,112,3)
    img4_stack = tf.stack([imglis_4[i] for i in range(594)])
#     print(img4_stack.shape)  # (594,112,112,3)
    img5_stack = tf.stack([imglis_5[i] for i in range(416)])
#     print(img5_stack.shape)  # (416,112,112,3)
    img6_stack = tf.stack([imglis_6[i] for i in range(626)])
#     print(img6_stack.shape)  # (626,112,112,3)
    feature_map = tf.concat([img0_stack,img1_stack,img2_stack,img3_stack,img4_stack,img5_stack,img6_stack], axis=0)
#     print(feature_map.shape)  # (3589,,112,112,3)

    emotions = {
        '0': 'anger',  # 生气
        '1': 'disgust',  # 厌恶
        '2': 'fear',  # 恐惧
        '3': 'happy',  # 开心
        '4': 'sad',  # 伤心
        '5': 'surprised',  # 惊讶
        '6': 'normal',  # 中性
    }
    _label_0 = tf.zeros([491])
    _label_1 = tf.ones([55])
    _label_2 = tf.ones([528]) + tf.ones([528])
    _label_3 = tf.ones([879]) + tf.ones([879]) + tf.ones([879])
    _label_4 = tf.ones([594]) + tf.ones([594]) + tf.ones([594]) + tf.ones([594])
    _label_5 = tf.ones([416]) + tf.ones([416]) + tf.ones([416]) + tf.ones([416]) + tf.ones([416])
    _label_6 = tf.ones([626]) + tf.ones([626]) + tf.ones([626]) + tf.ones([626]) + tf.ones([626]) + tf.ones([626])
    label = tf.concat([_label_0, _label_1, _label_2,_label_3,_label_4,_label_5,_label_6], axis=0)
#     print(label.shape) # (3589,)

    return feature_map, label

def load_valid_data():
    imglis_0 = valid_0()  # (467,112,112,3)
    imglis_1 = valid_1()  # (56,112,112,3)
    imglis_2 = valid_2() # (496,112,112,3)
    imglis_3 = valid_3()  # (895,112,112,3)
    imglis_4 = valid_4() # (653,112,112,3)
    imglis_5 = valid_5()  # (415,112,112,3)
    imglis_6 = valid_6() # (607,112,112,3)

    img0_stack = tf.stack([imglis_0[i] for i in range(467)])
#     print(img0_stack.shape)  # (467,112,112,3)
    img1_stack = tf.stack([imglis_1[i] for i in range(56)])
#     print(img1_stack.shape)  # (56,112,112,3)
    img2_stack = tf.stack([imglis_2[i] for i in range(496)])
#     print(img2_stack.shape)  # (496,112,112,3)
    img3_stack = tf.stack([imglis_3[i] for i in range(895)])
#     print(img3_stack.shape)  # (879,112,112,3)
    img4_stack = tf.stack([imglis_4[i] for i in range(653)])
#     print(img4_stack.shape)  # (594,112,112,3)
    img5_stack = tf.stack([imglis_5[i] for i in range(415)])
#     print(img5_stack.shape)  # (416,112,112,3)
    img6_stack = tf.stack([imglis_6[i] for i in range(607)])
#     print(img6_stack.shape)  # (626,112,112,3)
    feature_map = tf.concat([img0_stack,img1_stack,img2_stack,img3_stack,img4_stack,img5_stack,img6_stack], axis=0)
#     print(feature_map.shape)  # (3589,,112,112,3)

    emotions = {
        '0': 'anger',  # 生气
        '1': 'disgust',  # 厌恶
        '2': 'fear',  # 恐惧
        '3': 'happy',  # 开心
        '4': 'sad',  # 伤心
        '5': 'surprised',  # 惊讶
        '6': 'normal',  # 中性
    }
    _label_0 = tf.zeros([467])
    _label_1 = tf.ones([56])
    _label_2 = tf.ones([496]) + tf.ones([496])
    _label_3 = tf.ones([895]) + tf.ones([895]) + tf.ones([895])
    _label_4 = tf.ones([653]) + tf.ones([653]) + tf.ones([653]) + tf.ones([653])
    _label_5 = tf.ones([415]) + tf.ones([415]) + tf.ones([415]) + tf.ones([415]) + tf.ones([415])
    _label_6 = tf.ones([607]) + tf.ones([607]) + tf.ones([607]) + tf.ones([607]) + tf.ones([607]) + tf.ones([607])
    label = tf.concat([_label_0, _label_1, _label_2,_label_3,_label_4,_label_5,_label_6], axis=0)
#     print(label.shape) # (3589,)

    return feature_map, label

def load_train_data():
    imglis_0 = train_0()  # (3995,112,112,3)
    imglis_1 = train_1()  # (436,112,112,3)
    imglis_2 = train_2()  # (4097,112,112,3)
    imglis_3 = train_3()  # (7215,112,112,3)
    imglis_4 = train_4()  # (4830,112,112,3)
    imglis_5 = train_5()  # (3171,112,112,3)
    imglis_6 = train_6()  # (4965,112,112,3)

    img0_stack = tf.stack([imglis_0[i] for i in range(3995)])
#     print(img0_stack.shape)  # (3995,112,112,3)
    img1_stack = tf.stack([imglis_1[i] for i in range(436)])
#     print(img1_stack.shape)  # (436,112,112,3)
    img2_stack = tf.stack([imglis_2[i] for i in range(4097)])
#     print(img2_stack.shape)  # (4097,112,112,3)
    img3_stack = tf.stack([imglis_3[i] for i in range(7215)])
#     print(img3_stack.shape)  # (7215,112,112,3)
    img4_stack = tf.stack([imglis_4[i] for i in range(4830)])
#     print(img4_stack.shape)  # (4830,112,112,3)
    img5_stack = tf.stack([imglis_5[i] for i in range(3171)])
#     print(img5_stack.shape)  # (3171,112,112,3)
    img6_stack = tf.stack([imglis_6[i] for i in range(4965)])
#     print(img6_stack.shape)  # (4965,112,112,3)
    feature_map = tf.concat([img0_stack,img1_stack,img2_stack,img3_stack,img4_stack,img5_stack,img6_stack], axis=0)
#     print(feature_map.shape)  # (28709,112,112,3)

    emotions = {
        '0': 'anger',  # 生气
        '1': 'disgust',  # 厌恶
        '2': 'fear',  # 恐惧
        '3': 'happy',  # 开心
        '4': 'sad',  # 伤心
        '5': 'surprised',  # 惊讶
        '6': 'normal',  # 中性
    }
    _label_0 = tf.zeros([3995])
    _label_1 = tf.ones([436])
    _label_2 = tf.ones([4097]) + tf.ones([4097])
    _label_3 = tf.ones([7215]) + tf.ones([7215]) + tf.ones([7215])
    _label_4 = tf.ones([4830]) + tf.ones([4830]) + tf.ones([4830]) + tf.ones([4830])
    _label_5 = tf.ones([3171]) + tf.ones([3171]) + tf.ones([3171]) + tf.ones([3171]) + tf.ones([3171])
    _label_6 = tf.ones([4965]) + tf.ones([4965]) + tf.ones([4965]) + tf.ones([4965]) + tf.ones([4965]) + tf.ones([4965])
    label = tf.concat([_label_0, _label_1, _label_2,_label_3,_label_4,_label_5,_label_6], axis=0)
#     print(label.shape) # (28709,)

    return feature_map, label


if __name__ == '__main__':
    load_test_data()
    load_valid_data()
    load_train_data()