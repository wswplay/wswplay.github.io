---
titile: Nodejs
# outline: deep
---

# Nodejs

## fs 模块

### fs.existsSync(path)

如果路径存在则返回 true，否则返回 false。

```js
import { existsSync } from "node:fs";

if (existsSync("/etc/passwd")) console.log("The path exists.");
```

### fs.mkdirSync(path[, options])

同步地创建目录。 返回 undefined 或创建的第一个目录路径（如果 recursive 为 true）。 这是 fs.mkdir() 的同步版本。

### fs.readdirSync(path[, options])

同步读取**目录**的内容。

### fs.readFileSync(path[, options])

同步读取**文件**的内容。

### fs.writeFileSync(file, data[, options])

将 data 写入到 file。返回 undefined。

### fs.statSync(path[, options])

获取路径的文件内容信息。

### fs.copyFileSync(src, dest[, mode])

同步地复制 src 到 dest。

## fs-extra(fs 扩展)

### fs.pathExistsSync(path: string): boolean

`fs.existsSync(path)` 的别名。如果路径存在则返回 true，否则返回 false。

### fs.pathExists(path: string)

异步。返回值为 `Promise<boolean>`。如果路径存在则返回 true，否则返回 false。

## url 模块

### url.fileURLToPath(url)

此函数可确保正确解码百分比编码字符，并确保跨平台有效的绝对路径字符串。

```js
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);

new URL("file:///C:/path/").pathname; // 错误: /C:/path/
fileURLToPath("file:///C:/path/"); // 正确: C:\path\ (Windows)
```
