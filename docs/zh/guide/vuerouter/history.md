---
tittle: History
---
## History
```js
window.history.back();
window.history.forward();
window.history.go(-1);
window.history.go(1);
```
HTML5引入了 ```history.pushState()``` 和 ```history.replaceState()``` 方法，它们分别可以添加和修改历史记录条目。这些方法通常与 ```window.onpopstate``` 配合使用。但它们并不会导致加载刷新浏览器，只是改变了url。