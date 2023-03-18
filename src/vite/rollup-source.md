---
title: Rollup.js源码分析
outline: deep
---

# Rollup.js 源码摘要

[Rollup.js Github 地址](https://github.com/rollup/rollup)

调试命令：

```bash
# 构建
pnpm run build
# 进入调试流程
./dist/bin/rollup -c zhi.config.ts --configPlugin typescript
```

> **看源码技能 get :white_check_mark:**

- 1、注意 class 类，尤其是属性；
- 2、注意方法名称，代表功能；
- 3、看函数返回了什么，返回值才是目的嘛；

## 流程简介

主要分为两个大阶段：

- 1、打包 `build` 阶段。
- 2、输出 `generate`(如 ts 配置文件) 或者 写入 `write` 阶段。  
  generate 意思是，只输出在内存中，比如 `ts` 配置文件，会被先打包成 `.mjs` 文件，再读取配置内容，随后立即删除这个临时文件。  
  write 就是真正写入磁盘，变成看得见的实体文件。所有目标模块，都写入成实体文件。

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

## 打包

### rollupInternal() 执行 options 钩子

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
async function getInputOptions() {
  const { options, unsetOptions } = await normalizeInputOptions(
		await rawPlugins.reduce(applyOptionHook(watchMode), Promise.resolve(rawInputOptions))
	);
}
// 执行options钩子
function applyOptionHook(watchMode: boolean) {
	return async (inputOptions: Promise<RollupOptions>, plugin: Plugin): Promise<InputOptions> => {
		const handler = 'handler' in plugin.options! ? plugin.options.handler : plugin.options!;
		return (
			(await handler.call({ meta: { rollupVersion, watchMode } }, await inputOptions)) ||
			inputOptions
		);
	};
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
    // 解析入口文件所有模块，执行 resolveId 钩子
    ({ entryModules: this.entryModules, implicitEntryModules: this.implicitEntryModules } =
        await this.moduleLoader.addEntryModules(normalizeEntryModules(this.options.input), true));
    // 将模块信息分组保存：内部模块和外部模块
    for (const module of this.modulesById.values()) {
      if (module instanceof Module) {
        this.modules.push(module);
      } else {
        // 外部模块
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

```ts
const inputOptions = {
  external: (id: string) =>
    (id[0] !== "." && !isAbsolute(id)) || id.slice(-5, id.length) === ".json",
  input: fileName,
  onwarn: warnings.add,
  plugins: [],
  treeshake: false,
};
export async function normalizeInputOptions(config: InputOptions): Promise<{
  options: NormalizedInputOptions;
  unsetOptions: Set<string>;
}> {
  // These are options that may trigger special warnings or behaviour later
  // if the user did not select an explicit value
  const unsetOptions = new Set<string>();

  const context = config.context ?? "undefined";
  const onwarn = getOnwarn(config);
  const strictDeprecations = config.strictDeprecations || false;
  const maxParallelFileOps = getmaxParallelFileOps(
    config,
    onwarn,
    strictDeprecations
  );
  const options: NormalizedInputOptions & InputOptions = {
    acorn: getAcorn(config) as unknown as NormalizedInputOptions["acorn"],
    acornInjectPlugins: getAcornInjectPlugins(config),
    cache: getCache(config),
    context,
    experimentalCacheExpiry: config.experimentalCacheExpiry ?? 10,
    experimentalLogSideEffects: config.experimentalLogSideEffects || false,
    external: getIdMatcher(config.external),
    inlineDynamicImports: getInlineDynamicImports(
      config,
      onwarn,
      strictDeprecations
    ),
    input: getInput(config),
    makeAbsoluteExternalsRelative:
      config.makeAbsoluteExternalsRelative ?? "ifRelativeSource",
    manualChunks: getManualChunks(config, onwarn, strictDeprecations),
    maxParallelFileOps,
    maxParallelFileReads: maxParallelFileOps,
    moduleContext: getModuleContext(config, context),
    onwarn,
    perf: config.perf || false,
    plugins: await normalizePluginOption(config.plugins),
    preserveEntrySignatures: config.preserveEntrySignatures ?? "exports-only",
    preserveModules: getPreserveModules(config, onwarn, strictDeprecations),
    preserveSymlinks: config.preserveSymlinks || false,
    shimMissingExports: config.shimMissingExports || false,
    strictDeprecations,
    treeshake: getTreeshake(config),
  };
  return { options, unsetOptions };
}
async function mergeInputOptions() {
  const inputOptions: CompleteInputOptions<keyof InputOptions> = {
    acorn: getOption("acorn"),
    acornInjectPlugins: config.acornInjectPlugins as
      | (() => unknown)[]
      | (() => unknown)
      | undefined,
    cache: config.cache as false | RollupCache | undefined,
    context: getOption("context"),
    experimentalCacheExpiry: getOption("experimentalCacheExpiry"),
    experimentalLogSideEffects: getOption("experimentalLogSideEffects"),
    external: getExternal(config, overrides),
    inlineDynamicImports: getOption("inlineDynamicImports"),
    input: getOption("input") || [],
    makeAbsoluteExternalsRelative: getOption("makeAbsoluteExternalsRelative"),
    manualChunks: getOption("manualChunks"),
    maxParallelFileOps: getOption("maxParallelFileOps"),
    maxParallelFileReads: getOption("maxParallelFileReads"),
    moduleContext: getOption("moduleContext"),
    onwarn: getOnWarn(config, defaultOnWarnHandler),
    perf: getOption("perf"),
    plugins: await normalizePluginOption(config.plugins),
    preserveEntrySignatures: getOption("preserveEntrySignatures"),
    preserveModules: getOption("preserveModules"),
    preserveSymlinks: getOption("preserveSymlinks"),
    shimMissingExports: getOption("shimMissingExports"),
    strictDeprecations: getOption("strictDeprecations"),
    treeshake: getObjectOption(
      config,
      overrides,
      "treeshake",
      objectifyOptionWithPresets(
        treeshakePresets,
        "treeshake",
        URL_TREESHAKE,
        "false, true, "
      )
    ),
    watch: getWatch(config, overrides),
  };
}
async function mergeOutputOptions(
  config: OutputOptions,
  overrides: OutputOptions,
  warn: WarningHandler
): Promise<OutputOptions> {
  const getOption = (name: keyof OutputOptions): any =>
    overrides[name] ?? config[name];
  const outputOptions: CompleteOutputOptions<keyof OutputOptions> = {
    amd: getObjectOption(config, overrides, "amd"),
    assetFileNames: getOption("assetFileNames"),
    banner: getOption("banner"),
    chunkFileNames: getOption("chunkFileNames"),
    compact: getOption("compact"),
    dir: getOption("dir"),
    dynamicImportFunction: getOption("dynamicImportFunction"),
    dynamicImportInCjs: getOption("dynamicImportInCjs"),
    entryFileNames: getOption("entryFileNames"),
    esModule: getOption("esModule"),
    experimentalDeepDynamicChunkOptimization: getOption(
      "experimentalDeepDynamicChunkOptimization"
    ),
    experimentalMinChunkSize: getOption("experimentalMinChunkSize"),
    exports: getOption("exports"),
    extend: getOption("extend"),
    externalImportAssertions: getOption("externalImportAssertions"),
    externalLiveBindings: getOption("externalLiveBindings"),
    file: getOption("file"),
    footer: getOption("footer"),
    format: getOption("format"),
    freeze: getOption("freeze"),
    generatedCode: getObjectOption(
      config,
      overrides,
      "generatedCode",
      objectifyOptionWithPresets(
        generatedCodePresets,
        "output.generatedCode",
        URL_OUTPUT_GENERATEDCODE,
        ""
      )
    ),
    globals: getOption("globals"),
    hoistTransitiveImports: getOption("hoistTransitiveImports"),
    indent: getOption("indent"),
    inlineDynamicImports: getOption("inlineDynamicImports"),
    interop: getOption("interop"),
    intro: getOption("intro"),
    manualChunks: getOption("manualChunks"),
    minifyInternalExports: getOption("minifyInternalExports"),
    name: getOption("name"),
    namespaceToStringTag: getOption("namespaceToStringTag"),
    noConflict: getOption("noConflict"),
    outro: getOption("outro"),
    paths: getOption("paths"),
    plugins: await normalizePluginOption(config.plugins),
    preferConst: getOption("preferConst"),
    preserveModules: getOption("preserveModules"),
    preserveModulesRoot: getOption("preserveModulesRoot"),
    sanitizeFileName: getOption("sanitizeFileName"),
    sourcemap: getOption("sourcemap"),
    sourcemapBaseUrl: getOption("sourcemapBaseUrl"),
    sourcemapExcludeSources: getOption("sourcemapExcludeSources"),
    sourcemapFile: getOption("sourcemapFile"),
    sourcemapIgnoreList: getOption("sourcemapIgnoreList"),
    sourcemapPathTransform: getOption("sourcemapPathTransform"),
    strict: getOption("strict"),
    systemNullSetters: getOption("systemNullSetters"),
    validate: getOption("validate"),
  };
}
export async function normalizeOutputOptions() {
  const outputOptions: NormalizedOutputOptions & OutputOptions = {
    amd: getAmd(config),
    assetFileNames: config.assetFileNames ?? "assets/[name]-[hash][extname]",
    banner: getAddon(config, "banner"),
    chunkFileNames: config.chunkFileNames ?? "[name]-[hash].js",
    compact,
    dir: getDir(config, file),
    dynamicImportFunction: getDynamicImportFunction(
      config,
      inputOptions,
      format
    ),
    dynamicImportInCjs: config.dynamicImportInCjs ?? true,
    entryFileNames: getEntryFileNames(config, unsetOptions),
    esModule: config.esModule ?? "if-default-prop",
    experimentalDeepDynamicChunkOptimization:
      getExperimentalDeepDynamicChunkOptimization(config, inputOptions),
    experimentalMinChunkSize: config.experimentalMinChunkSize || 0,
    exports: getExports(config, unsetOptions),
    extend: config.extend || false,
    externalImportAssertions: config.externalImportAssertions ?? true,
    externalLiveBindings: config.externalLiveBindings ?? true,
    file,
    footer: getAddon(config, "footer"),
    format,
    freeze: config.freeze ?? true,
    generatedCode,
    globals: config.globals || {},
    hoistTransitiveImports: config.hoistTransitiveImports ?? true,
    indent: getIndent(config, compact),
    inlineDynamicImports,
    interop: getInterop(config),
    intro: getAddon(config, "intro"),
    manualChunks: getManualChunks(
      config,
      inlineDynamicImports,
      preserveModules,
      inputOptions
    ),
    minifyInternalExports: getMinifyInternalExports(config, format, compact),
    name: config.name,
    namespaceToStringTag: getNamespaceToStringTag(
      config,
      generatedCode,
      inputOptions
    ),
    noConflict: config.noConflict || false,
    outro: getAddon(config, "outro"),
    paths: config.paths || {},
    plugins: await normalizePluginOption(config.plugins),
    preferConst,
    preserveModules,
    preserveModulesRoot: getPreserveModulesRoot(config),
    sanitizeFileName:
      typeof config.sanitizeFileName === "function"
        ? config.sanitizeFileName
        : config.sanitizeFileName === false
        ? (id) => id
        : defaultSanitizeFileName,
    sourcemap: config.sourcemap || false,
    sourcemapBaseUrl: getSourcemapBaseUrl(config),
    sourcemapExcludeSources: config.sourcemapExcludeSources || false,
    sourcemapFile: config.sourcemapFile,
    sourcemapIgnoreList:
      typeof config.sourcemapIgnoreList === "function"
        ? config.sourcemapIgnoreList
        : config.sourcemapIgnoreList === false
        ? () => false
        : (relativeSourcePath) => relativeSourcePath.includes("node_modules"),
    sourcemapPathTransform: config.sourcemapPathTransform as
      | SourcemapPathTransformOption
      | undefined,
    strict: config.strict ?? true,
    systemNullSetters: config.systemNullSetters ?? true,
    validate: config.validate || false,
  };
  return { options: outputOptions, unsetOptions };
}
```

### 插件上下文

```ts
export function getPluginContext(
  plugin: Plugin,
  pluginCache: Record<string, SerializablePluginCache> | void,
  graph: Graph,
  options: NormalizedInputOptions,
  fileEmitter: FileEmitter,
  existingPluginNames: Set<string>
): PluginContext {
  ...
  return {
    addWatchFile(id) {
      if (graph.phase >= BuildPhase.GENERATE) {
        return this.error(errorInvalidRollupPhaseForAddWatchFile());
      }
      graph.watchFiles[id] = true;
    },
    cache: cacheInstance,
    emitFile: fileEmitter.emitFile.bind(fileEmitter),
    error(error_): never {
      return error(errorPluginError(error_, plugin.name));
    },
    getFileName: fileEmitter.getFileName,
    getModuleIds: () => graph.modulesById.keys(),
    getModuleInfo: graph.getModuleInfo,
    getWatchFiles: () => Object.keys(graph.watchFiles),
    load(resolvedId) {
      return graph.moduleLoader.preloadModule(resolvedId);
    },
    meta: {
      rollupVersion,
      watchMode: graph.watchMode,
    },
    get moduleIds() {
      function* wrappedModuleIds() {
        // We are wrapping this in a generator to only show the message once we are actually iterating
        warnDeprecation(
          `Accessing "this.moduleIds" on the plugin context by plugin ${plugin.name} is deprecated. The "this.getModuleIds" plugin context function should be used instead.`,
          URL_THIS_GETMODULEIDS,
          true,
          options,
          plugin.name
        );
        yield* moduleIds;
      }

      const moduleIds = graph.modulesById.keys();
      return wrappedModuleIds();
    },
    parse: graph.contextParse.bind(graph),
    resolve(
      source,
      importer,
      { assertions, custom, isEntry, skipSelf } = BLANK
    ) {
      return graph.moduleLoader.resolveId(
        source,
        importer,
        custom,
        isEntry,
        assertions || EMPTY_OBJECT,
        skipSelf ? [{ importer, plugin, source }] : null
      );
    },
    setAssetSource: fileEmitter.setAssetSource,
    warn(warning) {
      if (typeof warning === "string") warning = { message: warning };
      if (warning.code) warning.pluginCode = warning.code;
      warning.code = "PLUGIN_WARNING";
      warning.plugin = plugin.name;
      options.onwarn(warning);
    },
  };
}
```

## 输出

```md
rollupInternal
├── async write()
│ └── handleGenerateWrite()
├── src
│ ├── plugins
│ │ └── zhi-rollup-plugin.ts
│ └── xiao.ts
└── zhi.config.ts
```

```ts
// 开始输出函数 handleGenerateWrite
export async function rollupInternal() {
  const result: RollupBuild = {
    async generate(rawOutputOptions: OutputOptions) {...},
    async write(rawOutputOptions: OutputOptions) {
      if (result.closed) return error(errorAlreadyClosed());
      return handleGenerateWrite(
        true,
        inputOptions,
        unsetInputOptions,
        rawOutputOptions,
        graph
      );
    },
  };
  return result;
}
async function handleGenerateWrite() {
  // 获取输出配置项
  const { options: outputOptions } = await getOutputOptionsAndPluginDriver();
  return catchUnfinishedHookActions(outputPluginDriver, async () => {
    const bundle = new Bundle(outputOptions, unsetOptions, inputOptions, outputPluginDriver, graph);
    // 进入输出核心函数
    const generated = await bundle.generate(isWrite);
    // 如果是写入那就 writeOutputFile
    if (isWrite) {
      timeStart('WRITE', 1);
      if (!outputOptions.dir && !outputOptions.file) {
        return error(errorMissingFileOrDirOption());
      }
      await Promise.all(
        Object.values(generated).map(chunk =>
          graph.fileOperationQueue.run(() => writeOutputFile(chunk, outputOptions))
        )
      );
      await outputPluginDriver.hookParallel('writeBundle', [outputOptions, generated]);
      timeEnd('WRITE', 1);
    }
    return createOutput(generated);
  }
}
```

### bundle.generate()

```ts
export default class Bundle {
  async generate(isWrite: boolean): Promise<OutputBundle> {
    try {
      const outputBundleBase: OutputBundle = Object.create(null);
      // 执行 renderStart 钩子
      await this.pluginDriver.hookParallel("renderStart", [
        this.outputOptions,
        this.inputOptions,
      ]);
      // 生成 chunk
      const chunks = await this.generateChunks(
        outputBundle,
        getHashPlaceholder
      );
      for (const chunk of chunks) {
        chunk.generateExports();
      }
      await renderChunks(
        chunks,
        outputBundle,
        this.pluginDriver,
        this.outputOptions,
        this.inputOptions.onwarn
      );
    } catch (error_: any) {
      await this.pluginDriver.hookParallel("renderError", [error_]);
      throw error_;
    }
    // 执行 generateBundle 钩子
    await this.pluginDriver.hookSeq("generateBundle", [
      this.outputOptions,
      outputBundle as OutputBundle,
      isWrite,
    ]);
    // 返回基础信息包
    return outputBundleBase;
  }
}
```

### generateChunks

```ts
export default class Bundle {
  private async generateChunks() {
    const chunks: Chunk[] = [];
    for (const { alias, modules } of inlineDynamicImports) {
      const chunk = new Chunk(
        modules,
        this.inputOptions,
        this.outputOptions,
        this.unsetOptions,
        this.pluginDriver,
        this.graph.modulesById,
        chunkByModule,
        externalChunkByModule,
        this.facadeChunkByModule,
        this.includedNamespaces,
        alias,
        getHashPlaceholder,
        bundle,
        inputBase,
        snippets
      );
      chunks.push(chunk);
    }
    return [...chunks, ...facades];
  }
}
export async function renderChunks() {
  const renderedChunks = await Promise.all(
    chunks.map((chunk) => chunk.render())
  );
  const chunkGraph = getChunkGraph(chunks);
}
export default class Chunk {
  async render(): Promise<ChunkRenderResult> {
    const {
      accessedGlobals,
      indent,
      magicString,
      renderedSource,
      usedModules,
      usesTopLevelAwait,
    } = this.renderModules(preliminaryFileName.fileName);
    return {
      chunk: this,
      magicString,
      preliminaryFileName,
      usedModules,
    };
  }
  private renderModules(fileName: string) {
    const renderOptions: RenderOptions = {
      dynamicImportFunction,
      exportNamesByVariable,
      format,
      freeze,
      indent,
      namespaceToStringTag,
      pluginDriver,
      snippets,
      useOriginalName: null,
    };
    const rendered = module.render(renderOptions);
    const { renderedExports, removedExports } = module.getRenderedExports();
    return {
      accessedGlobals,
      indent,
      magicString,
      renderedSource,
      usedModules,
      usesTopLevelAwait,
    };
  }
}
```

## PS: 流程目录

<!--@include: ./extend/rollup-source-build.md-->
