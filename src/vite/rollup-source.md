---
title: Rollup.js源码分析
---

# Rollup.js 源码浅析

[Github 地址](https://github.com/rollup/rollup)

## rollup

```js
export default function rollup(rawInputOptions: RollupOptions): Promise<RollupBuild> {
	return rollupInternal(rawInputOptions, null);
}
export async function rollupInternal(
	rawInputOptions: RollupOptions,
	watcher: RollupWatcher | null
): Promise<RollupBuild> {

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
