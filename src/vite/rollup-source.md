---
title: Rollup.js源码分析
outline: deep
---

# Rollup.js 源码浅析

[Github 地址](https://github.com/rollup/rollup)

## rollup 命令源码

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
}
export default function rollup(
  rawInputOptions: RollupOptions
): Promise<RollupBuild> {
  return rollupInternal(rawInputOptions, null);
}
```

## 打包核心函数

### rollupInternal()

```ts
export default function rollup(rawInputOptions: RollupOptions): Promise<RollupBuild> {
	return rollupInternal(rawInputOptions, null);
}
async function rollupInternal(rawInputOptions, watcher) {
  // 执行options钩子
  const { options: inputOptions, unsetOptions: unsetInputOptions } = await getInputOptions(
		rawInputOptions,
		watcher !== null
	);
  const graph = new Graph(inputOptions, watcher);
  await catchUnfinishedHookActions(graph.pluginDriver, async () => {
    try {
      timeStart("initialize", 2);
      // 执行buildStart钩子
      await graph.pluginDriver.hookParallel("buildStart", [inputOptions]);
      timeEnd("initialize", 2);
      // 就是下面的
      await graph.build();
    } catch (error_) {
      const watchFiles = Object.keys(graph.watchFiles);
      if (watchFiles.length > 0) {
        error_.watchFiles = watchFiles;
      }
      await graph.pluginDriver.hookParallel("buildEnd", [error_]);
      await graph.pluginDriver.hookParallel("closeBundle", []);
      throw error_;
    }
    // 执行buildEnd钩子
    await graph.pluginDriver.hookParallel("buildEnd", []);
  });
}
async function catchUnfinishedHookActions(pluginDriver, callback) {
  const result = await Promise.race([callback(), emptyEventLoopPromise]);
}
async hookParallel(hookName, parameters, replaceContext) {
  const parallelPromises = [];
  for (const plugin of this.getSortedPlugins(hookName)) {
    if (plugin[hookName].sequential) {
      await Promise.all(parallelPromises);
      parallelPromises.length = 0;
      await this.runHook(hookName, parameters, plugin, replaceContext);
    }
    else {
      parallelPromises.push(this.runHook(hookName, parameters, plugin, replaceContext));
    }
  }
  await Promise.all(parallelPromises);
}
```

### async build()

```ts
// 上一步的 graph.build();
async build() {
  timeStart('generate module graph', 2);
  // 生成模块图谱
  await this.generateModuleGraph();
  timeEnd('generate module graph', 2);
  timeStart('sort and bind modules', 2);
  this.phase = BuildPhase.ANALYSE;
  this.sortModules();
  timeEnd('sort and bind modules', 2);
  timeStart('mark included statements', 2);
  this.includeStatements();
  timeEnd('mark included statements', 2);
  this.phase = BuildPhase.GENERATE;
}
```
