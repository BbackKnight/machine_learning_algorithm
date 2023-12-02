#! /usr/bin/env python
# -*- coding:utf-8 -*-
# @Time: 2023/12/210:47
# @File: fourier_1.PY
"""
循环的方式实现DTFT(离散傅里叶变换)
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import random
from mpl_toolkits.mplot3d import Axes3D

# 手动创建一个信号，由两个sin函数组合，频率幅值分别是[4, 2.5]、[6.5, 1.5]
srate = 1000  # 采样率，hz，即每秒采样1000次
time = np.arange(0, 2, 1 / srate)  # 时间序列对2秒按0.1步长分割，x轴坐标[0.1, 0.2, 0.3, ....2]
pnts = len(time)  # 绘制的点等于时间序列的长度
signal = 2.5 * np.sin(2 * np.pi * 4 * time) + 1.5 * np.sin(2 * np.pi * 6.5 * time)

# 准备傅里叶变换的数据
fourier_time = np.array(range(pnts)) / pnts
f_coefs = np.zeros(len(signal), dtype=complex)

# 时域信号图
plt.plot(time, signal)
plt.xlabel('Time (sec.)')
plt.ylabel('Amplitude (a.u.)')
plt.title('Signal')
plt.show()

# range(pnts):[0,1,2,3,....N]
# 我们并不知道原始的信号由哪些频率分量，那我们穷举，通过傅里叶变换，把很大范围的频率都求出来
# 不为0的信号分量就是有效成分

for fi in range(pnts):
    csw = np.exp(-1j * 2 * np.pi * fi * fourier_time)  # 创建复数形式的三角函数 exp(-i2π * k * t)，习惯称为基
    # 将原信号和该基相乘求和，除以pnts，是因为在离散傅里叶变换中，基的模不为1，为了归一化处理
    f_coefs[fi] = np.sum(np.multiply(signal, csw)) / pnts

# fcoefs是复数形式，取模，得到幅值，* 2后面会解释，是因为信号的频率关于y轴对称，但我们只求了正频率
# 如果不能理解为啥 * 2 ，可以先放一边，或者就记住也行，不影响对整体的理解
ampls = 2 * np.abs(f_coefs)

# 生成频率值，用于x轴，srate/2表示，只取采样频率的一半，根据奈奎斯特采样定律，只有一半的采样频率是正确的
hz = np.linspace(0, srate / 2, int(math.floor(pnts / 2.) + 1))
# 在x-y平面绘制频谱，
plt.stem(hz, ampls[range(len(hz))])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (a.u.)')
plt.xlim(0, 10)
plt.show()
