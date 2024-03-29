# python 2
#coding=utf-8
import datetime
import os
import h5py
import numpy as np

# f = h5py.File('path/filename.h5','r') #打开h5文件
f = h5py.File('E:/2018/AudioSet/bal_train.h5','r')
f.keys() #可以查看所有的主键
print([key for key in f.keys()])
————————————————
版权声明：本文为CSDN博主「Eve_if」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/Eve_if/article/details/84591493