---
title: Nodejs-url模块及扩展介绍与使用
---

# url 模块及扩展

## url 模块

### url.fileURLToPath(url)

此函数可确保正确解码百分比编码字符，并确保跨平台有效的绝对路径字符串。

```js
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);

new URL("file:///C:/path/").pathname; // 错误: /C:/path/
fileURLToPath("file:///C:/path/"); // 正确: C:\path\ (Windows)
```
