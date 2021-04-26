---
title: three.js
---
## 创建场景对象
```js
var scene = new THREE.Scene();
```
## 创建几何体对象
```js
// 创建几何对象，如一个立方体
var geometry = new THREE.BoxGeometry(100, 100, 100);
var material = new THREE.MeshLambertMaterial({
  color: 0xff0099,
  // vertexColors: THREE.VertexColors,
}); //材质对象Material
var mesh = new THREE.Mesh(geometry, material); //网格模型对象Mesh

//点对象添加到场景中
scene.add(mesh);
```
## 创建点光源
```js
var point = new THREE.PointLight(0xffffff);
point.position.set(150, 160, 160); //点光源位置
scene.add(point); //点光源添加到场景中
```
## 相机设置及创建相机对象
```js
// 相机设置
var width = window.innerWidth; //窗口宽度
var height = window.innerHeight; //窗口高度
var k = width / height; //窗口宽高比
var s = 162; //三维场景显示范围控制系数，系数越大，显示的范围越大
//创建相机对象
var camera = new THREE.OrthographicCamera(-s * k, s * k, s, -s, 1, 1000);
camera.position.set(100, 100, 200); //设置相机位置
camera.lookAt(scene.position); //设置相机方向(指向的场景对象)
```
## 创建渲染器对象
```js
var renderer = new THREE.WebGLRenderer();
renderer.setSize(width, height);//设置渲染区域尺寸
renderer.setClearColor(0xb9d3ff, 1); //设置背景颜色
document.body.appendChild(renderer.domElement); //body元素中插入canvas对象
```
## 执行渲染操作
```js
renderer.render(scene, camera);
```