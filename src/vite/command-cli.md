---
title: Vite命令源码解析
---

# Vite 命令源码解析

【[Github 地址](https://github.com/vitejs/vite/tree/main/packages/vite)】

## vite [dev/serve]

```ts
const { createServer } = await import("./server");
try {
  // 创建服务
  const server = await createServer({
    root,
    base: options.base,
    mode: options.mode,
    configFile: options.config,
    logLevel: options.logLevel,
    clearScreen: options.clearScreen,
    optimizeDeps: { force: options.force },
    server: cleanOptions(options),
  });
  // 监听服务
  await server.listen();

  const info = server.config.logger.info;

  const viteStartTime = global.__vite_start_time ?? false;
  const startupDurationString = viteStartTime
    ? colors.dim(
        `ready in ${colors.reset(
          colors.bold(Math.ceil(performance.now() - viteStartTime))
        )} ms`
      )
    : "";

  info(
    `\n  ${colors.green(
      `${colors.bold("VITE")} v${VERSION}`
    )}  ${startupDurationString}\n`,
    { clear: !server.config.logger.hasWarned }
  );
  // 打印信息
  server.printUrls();
}
```

## createServer

```ts
export async function createServer(
  inlineConfig: InlineConfig = {}
): Promise<ViteDevServer> {
  const config = await resolveConfig(inlineConfig, "serve");
  const container = await createPluginContainer(config, moduleGraph, watcher);
}
```

## 流程辅助函数

```ts
// 解析配置
export async function resolveConfig(...){
  if (configFile !== false) {
    // 加载配置文件
    const loadResult = await loadConfigFromFile(...)
    if (loadResult) {
      // 合并配置
      config = mergeConfig(loadResult.config, config)
      configFile = loadResult.path
      configFileDependencies = loadResult.dependencies
    }
  }
  // 排序分类插件
  const [prePlugins, normalPlugins, postPlugins] =
    sortUserPlugins(rawUserPlugins)
  // 处理服务配置
  const server = resolveServerOptions(resolvedRoot, config.server, logger)
}
export async function loadConfigFromFile(...){
  if (configFile) {
    // 如果入参指定了配置文件，就取路径
    resolvedPath = path.resolve(configFile)
  } else {
    // 否则，就地查询是否有 vite.config.js/mjs/ts/cjs/mts/cts 后缀文件
    for (const filename of DEFAULT_CONFIG_FILES) {
      const filePath = path.resolve(configRoot, filename)
      if (!fs.existsSync(filePath)) continue

      resolvedPath = filePath
      break
    }
  }
  // 找不到配置文件就提示报错信息
  if (!resolvedPath) {
    debug('no config file found.')
    return null
  }
  // 读取配置信息
  const userConfig = await loadConfigFromBundledFile(
    resolvedPath,
    bundled.code,
    isESM,
  )
  // 返回配置
  return {
    path: normalizePath(resolvedPath),
    config,
    dependencies: bundled.dependencies,
  }
}
// 执行config钩子
async function runConfigHook(...){
  let conf = config;
  for (const p of getSortedPluginsByHook("config", plugins)) {
    const hook = p.config;
    const handler = hook && "handler" in hook ? hook.handler : hook;
    if (handler) {
      const res = await handler(conf, configEnv);
      // 如果有返回值，就将其合并到配置中
      if (res) {
        conf = mergeConfig(conf, res);
      }
    }
  }
  return conf;
}
```
