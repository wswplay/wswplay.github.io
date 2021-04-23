---
title: 概念区分
---
## OpenGL
OpenGL（英语：Open Graphics Library，译名：开放图形库或者“开放式图形库”）是用于渲染2D、3D矢量图形的跨语言、跨平台的应用程序编程接口（API）。是一个底层框架。

### Unity3D
Unity3D 一般指Unity（游戏引擎）。Unity是实时3D互动内容创作和运营平台。包括游戏开发、美术、建筑、汽车设计、影视在内的所有创作者，借助Unity将创意变成现实。    
引擎里有图形绘制和3D渲染的需求，因此有一部分功能会通过OpenGL这个底层框架来实现。

## WebGL
WebGL（全写Web Graphics Library）是一种3D绘图协议，这种绘图技术标准允许把JavaScript和OpenGL ES 2.0结合在一起，通过增加OpenGL ES 2.0的一个JavaScript绑定，WebGL可以为HTML5 Canvas提供硬件3D加速渲染，这样Web开发人员就可以借助系统显卡来在浏览器里更流畅地展示3D场景和模型了，还能创建复杂的导航和数据视觉化。显然，WebGL技术标准免去了开发网页专用渲染插件的麻烦，可被用于创建具有复杂3D结构的网站页面，甚至可以用来设计3D网页游戏等等。

WebGL是包含OpenGL绘图标准，但是更为顶层。从图形学API的角度来看WebGL API是为突出个性而删减版的 OpenGL API。WebGL是OpenGL的一个子集。

:::tip

WebGL 2.0基于OpenGL ES 3.0，确保了提供许多选择性的WebGL 1.0扩展，并引入新的API。可利用部分Javascript实现自动存储器管理。 
:::

### three.js
拿[three.js](https://github.com/mrdoob/three.js)来类比，three.js是基于WebGL的3D框架，集成webgl所有应用特点，还把3D概念都打包成类库，只要用javascript直接调用three.js的类库，一样支持webgl的3D绘图标准。    
three.js是webgl的子集，却能够自己完成一个3D项目的开发，只要不是做技术研究，更为顶层的引擎无疑开发效率更高。

## WebVR API 
WebVR API 能为虚拟现实设备的渲染提供支持 — 例如像Oculus Rift或者HTC Vive 这样的头戴式设备与 Web apps 的连接。它能让开发者将位置和动作信息转换成3D场景中的运动。基于这项技术能产生很多有趣的应用, 比如虚拟的产品展示，可交互的培训课程，以及超强沉浸感的第一人称游戏。

## 着色器语言GLSL ES
着色器语言用于计算机图形编程，运行在GPU中，平时所说的大多数语言编写的程序都是运行在CPU中。 与OpenGL API相配合的是着色器语言GLSL，与OpenGL ES API、WebGL API相互配合的是着色器语言GLSL ES。**OpenGL标准应用的是客户端，OpenGL ES应用的是移动端，WebGL标准应用的是浏览器平台。**







### 问答
1. VR看房技术，哪家做得最好？众趣科技？还是美象、未来场景？    
以全球的角度来看，mp、众趣、如视这三家应该是第一序列的。但要说谁做的最好，应该是各有千秋吧。

#### 参考
1. [现在webAR做到了什么程度？](https://www.zhihu.com/question/301451219/answer/1029216255)
2. [Three.js中文文档](http://www.yanhuangxueyuan.com/threejs/docs/index.html)