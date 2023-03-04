---
title: Typescript配置文件tsconfig.json字段详解
outline: deep
---

# 配置文件 `tsconfig.json` 字段详解

每个字段，均有默认值。

## 举个例子

```json
{
  "compilerOptions": {
    "baseUrl": ".", // 设置解析非相对模块名称的基本目录，相对模块不会受 baseUrl 的影响
    "outDir": "dist", // 输出目录
    // 是否将 map 文件内容和 js 文件编译在同一个 js 文件中
    "sourceMap": false, // true：则map内容以//# sourceMappingURL= 然后接base64字符串插入js文件底部
    "target": "es2016", // 指定 ts 编译完之后的版本目标
    "newLine": "LF", // 指定发送文件时要使用的行尾序列：“CRLF”（dos）或“LF”（unix）
    "useDefineForClassFields": false,
    "module": "esnext", // // 指定使用的模块标准
    "moduleResolution": "node", // 用于选择模块解析策略，有 node 和 classic 两种类型
    "allowJs": false, // 是否允许编译 JS 文件，默认是 false，不编译
    "strict": true, // 否启动所有类型检查
    "noUnusedLocals": true, // 是否检查未使用的局部变量
    "experimentalDecorators": true, // 是否支持实验性装饰器
    "resolveJsonModule": true, // 是否允许导入json模块
    "esModuleInterop": true, // 第三方没有default导出的库中导入到es6模块(ts)中
    "removeComments": false, // 是否移除注释
    "jsx": "preserve", // 指定 JSX 的处理方式
    "lib": ["esnext", "dom"], // 指定要包含在编译中的库文件
    "types": ["jest", "puppeteer", "node"],
    "rootDir": ".", // 设置项目的根目录
    "paths": {
      // 设置模块名到基于 baseUrl 的路径映射
      "@vue/compat": ["packages/vue-compat/src"],
      "@vue/*": ["packages/*/src"],
      "vue": ["packages/vue/src"]
    },
    "allowSyntheticDefaultImports": true, // 指定允许从没有默认导出的模块中默认导入
    "forceConsistentCasingInFileNames": true, // 是否强制使用模块文件名必须和文件系统中文件名大小写一致
    "isolatedModules": true, // 是否将每个文件作为单独的模块，它不可以和 declaration 同时设定
    "noEmitOnError": true, // 是否在发生错误时禁止输出 JavaScript 代码
    "noUnusedParameters": true, // 是否检查未使用的参数
    "pretty": true, // 是否格式化输出的 JavaScript 代码
    "skipLibCheck": true // 是否跳过库声明文件检查
  },
  "include": ["src", "zhi.config.ts"], // 包括
  "exclude": ["dist", "node_modules"], // 排除
  "files": [], // 需要被编译的目标文件
  "extends": "@vue/tsconfig/tsconfig.web.json", // 继承
  "references": [{ "path": "./tsconfig.node.json" }] // 引用
}
```

## 字段详解

### compilerOptions

编译器的选项，如语言版本、目标 JavaScript 版本、生成的 sourcemap 等。

### exclude

exclude 表示要排除的、不编译的文件，可以指定一个列表，规则和 include 一样，可以是文件或文件夹，可以是相对路径或绝对路径，可以使用通配符。

### include

include 也可以指定要编译的路径列表。

与 files 区别：可以是**文件夹、文件、相对和绝对路径**，而且**可用通配符**。

### files：只能是文件

files 可配置文件路径数组，相对或绝对路径。编译器编译时候只会编译包含在 files 中的文件。

如不指定，则取决于有没有设置 include 选项，如无 include，则默认会编译根目录以及所有子目录中文件。

路径必须**是文件**，而**不是文件夹**，且 \* ? \*\*/ 等**通配符不能使用**。

### extends 继承与覆盖

extends 可指定一个其他的 `tsconfig.json` 文件路径，来继承这个配置文件里的配置，继承配置会**覆盖当前**文件配置。`TS 3.2` 开始，支持继承一个来自 `Node.js` 包的 `tsconfig.json` 配置文件。

### references 引用

一个对象数组，指定要引入的项目。可为项目不同部分使用不同 `ts` 配置。

使用 `references` 字段引入的配置文件需要设置 `composite: true` 字段，并用 `include 或 files` 等等属性指明配置覆盖的文件范围。

```json
// tsconfig.json
{
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

```json
// tsconfig.node.json
{
  "compilerOptions": {
    "composite": true
  },
  "include": ["vite.config.ts"]
}
```

### compileOnSave

如果设为 true，在我们编译了项目中文件保存的时候，编译器会根据 tsconfig.json 的配置重新生成文件(需要编辑器支持)。
