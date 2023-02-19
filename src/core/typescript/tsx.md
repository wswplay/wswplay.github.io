---
titile: tsx
---

# tsx：TS 文件执行器

众所周知，Node.js 并不支持直接执行 TS 文件，一般借助第三方才能执行。如之前的 ts-node，现在的 tsx。

`TypeScript Execute` 的缩写，出自 [esbuild](https://github.com/esbuild-kit) 门下。  
【[Github 库地址](https://github.com/esbuild-kit/tsx)】

## 安装

### 全局安装

```bash
npm install -g tsx
# 用法
tsx index.ts
```

### 局部安装

```bash
npm install --save-dev tsx
```

在 package.json 中用

```json
{
  "scripts": {
    "dev": "tsx ..."
  }
}
```

二进制文件，需要 [npx](/node/npx) 调用

```bash
npx tsx ...
```
