Otsu 阈值算法的简单实现

## 1. 什么是 Otsu 阈值算法

Otsu 阈值算法是一种自适应阈值分割算法，用于将图像分割为前景和背景。核心思想是通过最大化类间方差来确定最佳阈值。

Otsu 阈值算法的基本思想是：通过计算图像的灰度直方图，找到一个阈值，使得前景和背景的类间方差最大。

## 2. 算法步骤

1. 计算图像的灰度直方图:
    - 图像的像素值被划分为若干个灰度级，计算每个灰度级的像素数。
    - 计算每个灰度级的像素数占总像素数的比例。

2. 计算前景和背景的类间方差
    - 对每个可能的阈值，将图像的像素值划分为前景和背景。
    - 计算这两部分的灰度均值、像素比例和类间方差。
    - 寻找使类间方差最大的阈值。

## 3. 数学表达
- 假设总像素数为 $N$，灰度值的范围为 $[0, L-1]$，灰度级 $k$ 的像素数为 $n_k$，灰度级 $k$ 的像素数占总像素数的比例为 $p_k$。
- 对于x给定的阈值$t$，灰度值被分为两类：
    - 前景类$C_1$，灰度值在$[0, t]$之间
    - 背景类$C_2$，灰度值在$[t+1, L-1]$之间

- 前景类的像素数比例：$w_1 = \sum_{i=0}^{t} p_i$
- 背景类的像素数比例：$w_2 = \sum_{i=t+1}^{L-1} p_i$
- 前景类的灰度均值：$u_1 = \sum_{i=0}^{t} i p_i / w_1$
- 背景类的灰度均值：$u_2 = \sum_{i=t+1}^{L-1} i p_i / w_2$
- 类间方差：$\sigma_b^2 = w_1 w_2 (u_1 - u_2)^2$