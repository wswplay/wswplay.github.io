---
title: 光源
---
你查看文档SpotLight、DirectionalLight、环境光AmbientLight等光源对象都有一个共同的基类Light，光源Light也有一个基类Object3D。也就是说Threejs环境光、点光源等子类光源可以继承Light和Object3D两个父类的属性和方法。

## 环境光AmbientLight
## 点光源PointLight
## 平行光DirectionalLight
## 聚光源SpotLight
## 光照计算算法
Three.js渲染的时候光照计算还是比较复杂的，这里不进行深入介绍，只给大家说下光源颜色和网格模型Mesh颜色相乘的知识，如果你有兴趣可以学习计算机图形学或者WebGL教程。

Threejs在渲染的时候网格模型材质的颜色值mesh.material.color和光源的颜色值light.color会进行相乘，简单说就是RGB三个分量分别相乘。

平行光漫反射简单数学模型：漫反射光的颜色 = 网格模型材质颜色值 x 光线颜色 x 光线入射角余弦值

漫反射数学模型RGB分量表示：(R2,G2,B2) = (R1,G1,B1) x (R0,G0,B0) x cosθ

## 光照阴影实时计算
