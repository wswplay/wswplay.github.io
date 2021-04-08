---
title: 路由
---
## createRouter
push 和 replace 方法内部都是执行了 changeLocation 方法，该函数内部执行了浏览器底层的 history.pushState 或者 history.replaceState 方法，会向当前浏览器会话的历史堆栈中添加一个状态，这样就在不刷新页面的情况下修改了页面的 URL。

