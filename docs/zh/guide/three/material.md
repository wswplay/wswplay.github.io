---
title: 材质对象
---
## 着色器
所有材质就是对WebGL着色器代码的封装。

### 点材质PointsMaterial
### 线材质
线材质有基础线材质LineBasicMaterial和虚线材质LineDashedMaterial两个。

## 继承与属性
点材质PointsMaterial、基础线材质LineBasicMaterial、基础网格材质MeshBasicMaterial、高光网格材质MeshPhongMaterial等材质都是父类Material的子类。

### side属性
### 透明度opacity
.opacity属性值的范围是0.0~1.0，0.0值表示完全透明, 1.0表示完全不透明，.opacity默认值1.0。

当设置.opacity属性值的时候，需要设置材质属性transparent值为true，如果材质的transparent属性没设置为true, 材质会保持完全不透明状态。


