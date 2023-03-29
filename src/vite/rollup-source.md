---
title: Rollup.js源码分析
outline: deep
---

# Rollup.js 源码摘要

[Rollup.js Github 地址](https://github.com/rollup/rollup)

调试命令：

```bash
# 先构建
pnpm run build
# 进入调试流程
./dist/bin/rollup -c zhi.config.ts --configPlugin typescript
```

> **看源码技能 get :white_check_mark:**

- 1、注意 class 类，尤其是属性；
- 2、注意方法名称，代表功能；
- 3、看函数返回了什么，返回值才是目的嘛；

## rollup(命令) 源码摘要

- 获取参数。
- 解析配置。
- 唤起打包核心函数。

```ts
// 获取参数 cli/cli.ts
const command = argParser(process.argv.slice(2), {
  alias: commandAliases,
  configuration: { "camel-case-expansion": false },
});

if (command.help || (process.argv.length <= 2 && process.stdin.isTTY)) {
  console.log(`\n${help.replace("__VERSION__", version)}\n`);
} else if (command.version) {
  console.log(`rollup v${version}`);
} else {
  run(command);
}
// 即 run()
export default async function runRollup(
  command: Record<string, any>
): Promise<void> {
  try {
    // 解析配置
    const { options, warnings } = await getConfigs(command);
    try {
      for (const inputOptions of options) {
        // 开启打包流程
        await build(inputOptions, warnings, command.silent);
      }
    } catch (error: any) {
      warnings.flush();
      handleError(error);
    }
  } catch (error: any) {
    handleError(error);
  }
}
export default async function build(
  inputOptions: MergedRollupOptions,
  warnings: BatchWarnings,
  silent = false
): Promise<unknown> {
  // 唤起打包核心函数
  const bundle = await rollup(inputOptions as any);
  // 输出打包产物、写入目标文件
  await Promise.all(outputOptions.map(bundle.write));
  // 执行 closeBundle 钩子
  await bundle.close();
}
export default function rollup(
  rawInputOptions: RollupOptions
): Promise<RollupBuild> {
  return rollupInternal(rawInputOptions, null);
}
```

## 打包流程简介

主要分为 `2` 大阶段：

- 1、构建(`build`) 阶段。
- 2、输出(`write/generate`)阶段。  
  `generate` 意思是，只输出在内存中，比如 `ts` 配置文件，会被先打包成 `.mjs` 文件，再读取配置内容，随后立即删除这个临时文件。  
  `write` 就是真正写入磁盘，变成看得见的实体文件。所有目标模块，都写入成实体文件。

## 流程函数集锦

<!--@include: ./extend/rollup-source-build.md-->
<!--@include: ./extend/rollup-source-write.md-->
