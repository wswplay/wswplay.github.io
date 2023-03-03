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
  "include": ["src", "zhi.config.ts"],
  "exclude": ["dist", "node_modules"],
  "files": []
}
```

## 字段详解

### compilerOptions

编译器的选项，如语言版本、目标 JavaScript 版本、生成的 sourcemap 等。

### exclude

exclude 表示要排除的、不编译的文件，他也可以指定一个列表，规则和 include 一样，可以是文件或文件夹，可以是相对路径或绝对路径，可以使用通配符。

### include

include 也可以指定要编译的路径列表。但是和 files 的区别在于，这里的路径可以是文件夹，也可以是文件，可以使用相对和绝对路径，而且可以使用通配符。

### files

files 可以配置一个数组列表，里面包含指定文件的相对或绝对路径，编译器在编译的时候只会编译包含在 files 中列出的文件。
如果不指定，则取决于有没有设置 include 选项，如果没有 include 选项，则默认会编译根目录以及所有子目录中的文件。

这里列出的路径必须是指定文件，而不是某个文件夹，而且不能使用 \* ? \*\*/ 等通配符

### extends

extends 可以通过指定一个其他的 tsconfig.json 文件路径，来继承这个配置文件里的配置，继承来的文件配置会覆盖当前文件定义的配置。TS 在 3.2 版本开始，支持继承一个来自 Node.js 包的 tsconfig.json 配置文件。

### references

项目引用。一个对象数组，指定要引入的项目。允许用户为项目的不同部分使用不同的 TypeScript 配置。

```json
{
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

### compileOnSave

如果设为 true，在我们编译了项目中文件保存的时候，编译器会根据 tsconfig.json 的配置重新生成文件(需要编辑器支持)。
