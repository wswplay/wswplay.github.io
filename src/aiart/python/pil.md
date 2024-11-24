---
title: PIL、pillow
outline: deep
---

# Python Imaging Library

The Python Imaging Library (PIL), now known as Pillow, is a library that offers several standard procedures for manipulating images.

## convert()

PIL（Python Imaging Library）的 `convert` 方法用于在不同图像模式之间转换。常见的图像模式包括：

1. **"1" (1-bit pixels, black and white, stored with one pixel per byte)**:

   - 每个像素 1 位，仅有黑白两种颜色。

2. **"L" (8-bit pixels, grayscale)**:

   - 灰度图像，像素值范围是 0 到 255，0 表示黑色，255 表示白色。

3. **"P" (8-bit pixels, mapped to any other mode using a color palette)**:

   - 使用调色板的 8 位彩色图像，每个像素值是调色板中颜色的索引。

4. **"RGB" (3x8-bit pixels, true color)**:

   - 三通道彩色图像，分别是红、绿、蓝三个通道，每个通道的值范围是 0 到 255。

5. **"RGBA" (4x8-bit pixels, true color with transparency mask)**:

   - RGB 模式加上 Alpha 通道，支持透明度。

6. **"CMYK" (4x8-bit pixels, color separation)**:

   - 四通道彩色图像，分别是青（C）、品红（M）、黄（Y）和黑（K）四个通道，用于印刷用途。

7. **"YCbCr" (3x8-bit pixels, color video format)**:

   - 用于视频和数字图像的颜色模型，包括亮度（Y）、色度（Cb）和色差（Cr）。

8. **"LAB" (3x8-bit pixels, the L*a*b color space)**:

   - 表示色彩感知的 L*a*b\*模型，包含亮度(L)、红绿轴(a)和黄蓝轴(b)。

9. **"HSV" (3x8-bit pixels, Hue, Saturation, Value)**:

   - 色调（H）、饱和度（S）和亮度（V）的色彩模型。

10. **"I" (32-bit signed integer pixels)**:

- 32 位有符号整型图像，每个像素是一个整数值。

11. **"F" (32-bit floating point pixels)**:

- 32 位浮点型图像，像素值可以是浮点数。
