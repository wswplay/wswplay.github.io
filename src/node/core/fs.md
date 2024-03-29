---
title: Nodejs-fs-extra文件系统模块及扩展介绍与使用
outline: deep
---

# fs and fs-extra：文件系统与扩展

## fs：File system

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

### fs.mkdirSync(path)

同步地创建目录。

### fs.unlink/fsPromises.unlink

`fs.unlink(path, callback)` 和 `fsPromises.unlink(path)`。

异步地删除文件或符号链接。fs.unlink() 不适用于目录，无论是空目录还是其他目录。 要删除目录，请使用 fs.rmdir()。

```js
// callback 方式
import { unlink } from "node:fs";
unlink("/tmp/hello", (err) => {
  if (err) throw err;
  console.log("successfully deleted /tmp/hello");
});
// promises 方式
import { unlink } from "node:fs/promises";
try {
  await unlink("/tmp/hello");
  console.log("successfully deleted /tmp/hello");
} catch (error) {
  console.error("there was an error:", error.message);
}
```

## fs-extra(fs 扩展)

【[Github 地址](https://github.com/jprichardson/node-fs-extra)】

### fs.pathExistsSync(path: string): boolean

`fs.existsSync(path)` 的别名。如果路径存在则返回 true，否则返回 false。

### fs.pathExists(path: string)

异步。返回值为 `Promise<boolean>`。如果路径存在则返回 true，否则返回 false。
