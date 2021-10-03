---
title: Path
---
## Path.resolve()
:::tip
此方法，在拼装```Vue-Router```路径时，特别有用
:::
resolve 将路径或路径片段的序列解析为绝对路径。
```js
var path = require("path")     //引入node的path模块

path.resolve('/foo/bar', './baz')   // returns '/foo/bar/baz'
path.resolve('/foo/bar', 'baz')   // returns '/foo/bar/baz'
path.resolve('/foo/bar', '/baz')   // returns '/baz'
path.resolve('/foo/bar', '../baz')   // returns '/foo/baz'
path.resolve('home','/foo/bar', '../baz')   // returns '/foo/baz'
path.resolve('home','./foo/bar', '../baz')   // returns '当前工作目录/home/foo/baz'
path.resolve('home','foo/bar', '../baz')   // returns '当前工作目录/home/foo/baz'
```
怎么理解呢？上面的操作，其实相当于命令行中的 ```cd``` 操作，举例如下：
```bash
path.resolve('/foo/bar', '../baz')   # returns '/foo/baz'
# 相当于
cd /foo/bar
cd ..
cd baz
```
