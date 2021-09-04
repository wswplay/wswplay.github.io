---
title: 下载
---
一般后端会给一个供下载的api。那怎么创建被点击的实体呢？    
一般，我们创建一个a标签超链接，点击超链接访问文件，来实现下载。
```js
// url直接下载
function downALinkElement(url) {
  let a = document.createElement('a');
  a.href = url;
  a.style.display = 'none';
  document.body.appendChild(a).click();
  document.body.removeChild(a);
}
// 文件流下载：例如Blob
function downBlobFile (content, filename) {
  // 创建隐藏的可下载链接
  var eleLink = document.createElement('a');
  eleLink.download = filename;
  eleLink.style.display = 'none';
  // 字符内容转变成blob地址
  var blob = new Blob([content]);
  eleLink.href = window.URL.createObjectURL(blob);
  // 触发点击
  document.body.appendChild(eleLink);
  eleLink.click();
  // 然后移除
  document.body.removeChild(eleLink);
  // window.URL.revokeObjectURL(url); //释放掉blob对象
};
// Vue指令实现
```
