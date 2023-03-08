---
title: Rollup.js源码分析
outline: deep
---

# Rollup.js 源码摘要

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
  // 执行 closeBundle 钩子
  await bundle.close();
}
export default function rollup(
  rawInputOptions: RollupOptions
): Promise<RollupBuild> {
  return rollupInternal(rawInputOptions, null);
}
```

## 打包核心函数

### rollupInternal()

```ts{10,11}
export default function rollup(rawInputOptions: RollupOptions): Promise<RollupBuild> {
	return rollupInternal(rawInputOptions, null);
}
async function rollupInternal(rawInputOptions, watcher) {
  // 执行options钩子
  const { options: inputOptions, unsetOptions: unsetInputOptions } = await getInputOptions(
		rawInputOptions,
		watcher !== null
	);
  // 创建打包上下文实例(很重要，就是它了)
  const graph = new Graph(inputOptions, watcher);
  await catchUnfinishedHookActions(graph.pluginDriver, async () => {
    try {
      timeStart("initialize", 2);
      // 执行buildStart钩子
      await graph.pluginDriver.hookParallel("buildStart", [inputOptions]);
      timeEnd("initialize", 2);
      // 就是下面的 async build()
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

### class Graph

```ts
export default class Graph {
  constructor() {
    // 插件驱动
    this.pluginDriver = new PluginDriver();
    this.acornParser = acorn.Parser.extend();
    // 模块加载器
    this.moduleLoader = new ModuleLoader();
    // 文件操作队列
    this.fileOperationQueue = new Queue();
    this.pureFunctions = getPureFunctions(options);
  }
}
export class ModuleLoader {
  private async fetchModule() {
    // 创建模块实例
    const module = new Module();
    this.modulesById.set(id, module);
    this.graph.watchFiles[id] = true;
    const loadPromise: LoadModulePromise = this.addModuleSource(...).then(() => [
      this.getResolveStaticDependencyPromises(module),
      this.getResolveDynamicImportPromises(module),
      loadAndResolveDependenciesPromise,
    ]);
    this.moduleLoadPromises.set(module, loadPromise);
    const resolveDependencyPromises = await loadPromise;
    if (!isPreload) {
      await this.fetchModuleDependencies(module, ...resolveDependencyPromises);
    } else if (isPreload === RESOLVE_DEPENDENCIES) {
      await loadAndResolveDependenciesPromise;
    }
    return module;
  }
}
```

### async build()

```ts
// Graph.ts
export default class Graph {
  ...
  constructor() {...}
  // 上一步的 graph.build();
  async build() {
    // 进入加载、解析阶段：获取模块信息、引用依赖关系等，生成模块图谱谱系
    timeStart('generate module graph', 2);
    await this.generateModuleGraph();
    timeEnd('generate module graph', 2);

    // 进入分析阶段：排序、绑定
    timeStart('sort and bind modules', 2);
    this.phase = BuildPhase.ANALYSE;
    this.sortModules();
    timeEnd('sort and bind modules', 2);

    timeStart('mark included statements', 2);
    this.includeStatements();
    timeEnd('mark included statements', 2);

    // 进入生成阶段：
    this.phase = BuildPhase.GENERATE;
  }
  private async generateModuleGraph(): Promise<void> {
    // 解析获取入口模块，执行 resolveId 钩子
    ({ entryModules: this.entryModules, implicitEntryModules: this.implicitEntryModules } =
        await this.moduleLoader.addEntryModules(normalizeEntryModules(this.options.input), true));
    // 如没有入口模块，则提示报错信息
    if (this.entryModules.length === 0) {
			throw new Error('You must supply options.input to rollup');
		}
    // 将模块信息存入模块数组
    for (const module of this.modulesById.values()) {
			if (module instanceof Module) {
				this.modules.push(module);
			} else {
				this.externalModules.push(module);
			}
		}
  }
}
normalizeEntryModules(...)  // 返回规范化目标模块信息对象数组
// ModuleLoader.ts
export class ModuleLoader {
  ...
  constructor() {...}
  async addEntryModules() {
    const newEntryModules = await this.extendLoadModulesPromise(
      Promise.all(
        unresolvedEntryModules.map(({ id, importer }) =>
          this.loadEntryModule(id, true, importer, null)
        )
      )
    )
  }
  private async loadEntryModule() {
    const resolveIdResult = await resolveId(...)
  }
}
// resolveid.ts
export async function resolveId() {
  const pluginResult = await resolveIdViaPlugins(...)
}
```

### 执行 resolveId 钩子

```ts
export function resolveIdViaPlugins() {
  return pluginDriver.hookFirstAndGetPlugin(
    "resolveId",
    [source, importer, { assertions, custom: customOptions, isEntry }],
    replaceContext,
    skipped
  );
}
async hookFirstAndGetPlugin() {
  for (const plugin of this.getSortedPlugins(hookName)) {
    if (skipped?.has(plugin)) continue;
    const result = await this.runHook(hookName, parameters, plugin, replaceContext);
    if (result != null) return [result, plugin];
  }
  return null;
}
```

### 执行 load 钩子

```ts
// ModuleLoader.ts
export class ModuleLoader {
  private async addModuleSource() {
    try {
      source = await this.graph.fileOperationQueue.run(
        async () =>
          (await this.pluginDriver.hookFirst('load', [id])) ?? (await readFile(id, 'utf8'))
      );
    }
  }
}
```

### 执行 transform 钩子

```ts
// transform.ts
export default async function transform() {
  try {
	  code = await pluginDriver.hookReduceArg0('transform', ...)
  }
}
```

```ts
// Graph.ts
private sortModules(): void {
  const { orderedModules, cyclePaths } = analyseModuleExecution(this.entryModules);
}
```
