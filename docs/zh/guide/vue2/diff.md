---
title:  diff算法
---
**diff算法**，能是现实组件最小更新。那它到底是怎么实现的呢？
## Vue怎么判断节点是否相同
## 相同vonde
```js
function sameVnode (a, b) {
  return (
    a.key === b.key && (
      (
        a.tag === b.tag &&
        a.isComment === b.isComment &&
        isDef(a.data) === isDef(b.data) &&
        sameInputType(a, b)
      ) || (
        isTrue(a.isAsyncPlaceholder) &&
        a.asyncFactory === b.asyncFactory &&
        isUndef(b.asyncFactory.error)
      )
    )
  )
}
```
可以看出，如果有key，就判断tag等等之类的。
## 不同vonde
1. 生成新节点
2. 更新占位符vonde(父节点)
3. 删除旧节点