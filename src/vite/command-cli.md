---
title: vite命令源码解析
outline: deep
---

# vite 命令源码摘要

【[Vite Github 地址](https://github.com/vitejs/vite/tree/main/packages/vite)】

**调试步骤：**

- 1、`vite源码`在`vite项目`中路径为：`vite/packages/vite`。
- 2、在 `package.json` 中查看 `vite命令` 路径为：`bin/vite.js`。
- 3、`bin/vite.js` 中，从 `start()` 函数开始。实际是引入了 `node/cli.js`。

```ts
function start() {
  return import("../dist/node/cli.js");
}
```

- 4、对应找到 `src/node/cli.ts`，在合适位置打上断点，就可以调试了。

```bash
# 调试命令
tsx cli.ts --open --port 9000
```

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
  // 解析、获取全部配置
  const config = await resolveConfig(inlineConfig, "serve");
  // 获取根目录、服务器配置
  const { root, server: serverConfig } = config;
  // 解析文件watch配置
  const resolvedWatchOptions = resolveChokidarOptions(config, {
    disableGlobbing: true,
    ...serverConfig.watch,
  });
  // 声明中间件
  const middlewares = connect() as Connect.Server;
  // 创建http服务对象
  const httpServer = middlewareMode
    ? null
    : await resolveHttpServer(serverConfig, middlewares, httpsOptions);
  // 创建文件监听器，用于监听文件变动更新
  const watcher = chokidar.watch(
    path.resolve(root),
    resolvedWatchOptions
  ) as FSWatcher;
  // 初始化模块图谱，以存储模块信息及引用关系
  const moduleGraph: ModuleGraph = new ModuleGraph((url, ssr) =>
    container.resolveId(url, undefined, { ssr })
  );
  // 创建插件容器，返回Vite封装的特色钩子体系
  const container = await createPluginContainer(config, moduleGraph, watcher);
  // 服务实例真身
  const server: ViteDevServer = {
    config,
    middlewares,
    httpServer,
    watcher,
    pluginContainer: container,
    moduleGraph,
    // 监听服务
    async listen(port?: number, isRestart?: boolean) {
      // 开启服务，比如自动打开浏览器
      await startServer(server, port, isRestart);
      if (httpServer) {
        // 解析组装url链接 如http://localhost:3062/
        server.resolvedUrls = await resolveServerUrls(
          httpServer,
          config.server,
          config
        );
      }
      return server;
    },
    // 打印信息
    printUrls() {
      if (server.resolvedUrls) {
        printServerUrls(
          server.resolvedUrls,
          serverConfig.host,
          config.logger.info
        );
      } else if (middlewareMode) {
        throw new Error("cannot print server URLs in middleware mode.");
      } else {
        throw new Error(
          "cannot print server URLs before server.listen is called."
        );
      }
    },
  };
  // 收集transformIndexHtml钩子任务
  server.transformIndexHtml = createDevHtmlTransformFn(server);
  // 执行configureServer钩子
  const postHooks: ((() => void) | void)[] = [];
  for (const hook of config.getSortedPluginHooks("configureServer")) {
    postHooks.push(await hook(server));
  }
  // 执行一系列中间件
  // 执行中间件：是否支持跨域，默认支持
  const { cors } = serverConfig;
  if (cors !== false) {
    middlewares.use(corsMiddleware(typeof cors === "boolean" ? {} : cors));
  }
  // 执行中间件：代理设置
  const { proxy } = serverConfig;
  if (proxy) {
    middlewares.use(proxyMiddleware(httpServer, proxy, config));
  }
  // 执行configureServer 后置钩子
  postHooks.forEach((fn) => fn && fn());
  // 初始化服务
  let initingServer: Promise<void> | undefined;
  let serverInited = false;
  const initServer = async () => {
    if (serverInited) {
      return;
    }
    if (initingServer) {
      return initingServer;
    }
    initingServer = (async function () {
      // 执行buildStart钩子
      await container.buildStart({});
      if (isDepsOptimizerEnabled(config, false)) {
        // non-ssr
        await initDepsOptimizer(config, server);
      }
      initingServer = undefined;
      serverInited = true;
    })();
    return initingServer;
  };
  if (!middlewareMode && httpServer) {
    // overwrite listen to init optimizer before server start
    const listen = httpServer.listen.bind(httpServer);
    httpServer.listen = (async (port: number, ...args: any[]) => {
      try {
        await initServer();
      } catch (e) {
        httpServer.emit("error", e);
        return;
      }
      return listen(port, ...args);
    }) as any;
  } else {
    await initServer();
  }

  return server;
}
```

## 流程辅助函数

### 执行 config 系钩子(resolveConfig)

解析配置。执行 `config`，`configResolved` 钩子。

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
  // 插件分类、排序
  const [prePlugins, normalPlugins, postPlugins] =
    sortUserPlugins(rawUserPlugins)
  // 执行config钩子
  const userPlugins = [...prePlugins, ...normalPlugins, ...postPlugins]
  config = await runConfigHook(config, userPlugins, configEnv)
  // 解析服务配置
  const server = resolveServerOptions(resolvedRoot, config.server, logger)
  // 执行worker config钩子
  workerConfig = await runConfigHook(workerConfig, workerUserPlugins, configEnv)
  // 定义返回配置
  const resolved: ResolvedConfig = {
    ...config,
    ...resolvedConfig,
  }
  // 执行configResolved钩子
  await Promise.all([
    ...resolved
      .getSortedPluginHooks('configResolved')
      .map((hook) => hook(resolved)),
    ...resolvedConfig.worker
      .getSortedPluginHooks('configResolved')
      .map((hook) => hook(workerResolved)),
  ])
  // 返回处理完结的配置
  return resolved
}
```

### 解析设置 http 服务(resolveHttpServer)

```ts
export async function resolveHttpServer(
  { proxy }: CommonServerOptions,
  app: Connect.Server,
  httpsOptions?: HttpsServerOptions
): Promise<HttpServer> {
  if (!httpsOptions) {
    const { createServer } = await import("node:http");
    return createServer(app);
  }

  // #484 fallback to http1 when proxy is needed.
  if (proxy) {
    const { createServer } = await import("node:https");
    return createServer(httpsOptions, app);
  } else {
    const { createSecureServer } = await import("node:http2");
    return createSecureServer(
      {
        // Manually increase the session memory to prevent 502 ENHANCE_YOUR_CALM
        // errors on large numbers of requests
        maxSessionMemory: 1000,
        ...httpsOptions,
        allowHTTP1: true,
      },
      // @ts-expect-error TODO: is this correct?
      app
    ) as unknown as HttpServer;
  }
}
```

### 创建 Vite 钩子，执行 options 钩子

实现 rollup 插件上下文接口，执行 options 钩子，返回 Vite 钩子。

```ts
export async function createPluginContainer(
  config: ResolvedConfig,
  moduleGraph?: ModuleGraph,
  watcher?: FSWatcher
): Promise<PluginContainer> {
  // 引入rollup插件上下文
  import type { MinimalPluginContext, PluginContext as RollupPluginContext } from "rollup";
  const minimalContext: MinimalPluginContext = {
    meta: {
      rollupVersion,
      watchMode: true,
    },
  };
  function getModuleInfo(id: string) {
    const module = moduleGraph?.getModuleById(id);
    if (!module) {
      return null;
    }
    if (!module.info) {
      module.info = new Proxy(
        { id, meta: module.meta || EMPTY_OBJECT } as ModuleInfo,
        ModuleInfoProxy
      );
    }
    return module.info;
  }
  // 通过Context实现Rollupjs的PluginContext接口，为异步钩子创建上下文。
  // 官方注释如下：
  // we should create a new context for each async hook pipeline so that the
  // active plugin in that pipeline can be tracked in a concurrency-safe manner.
  // using a class to make creating new contexts more efficient
  class Context implements PluginContext {...}
  // 定义返回值
  const container: PluginContainer = {
    // options是异步立即执行函数
    options: await (async () => {
      let options = rollupOptions;
      for (const optionsHook of getSortedPluginHooks("options")) {
        // 执行options钩子
        options = (await optionsHook.call(minimalContext, options)) || options;
      }
      if (options.acornInjectPlugins) {
        parser = acorn.Parser.extend(
          ...(arraify(options.acornInjectPlugins) as any)
        );
      }
      return {
        acorn,
        acornInjectPlugins: [],
        ...options,
      };
    })(),
    // 获取模块信息功能函数
    getModuleInfo,
    async buildStart() {
      // 执行buildStart钩子
      await hookParallel(
        "buildStart",
        (plugin) => new Context(plugin),
        () => [container.options as NormalizedInputOptions]
      );
    },
    async resolveId(rawId, importer = join(root, 'index.html'), options) {...},
    async load(id, options) {...},
    async transform(code, id, options) {...},
    async close() {...}
  };
  return container;
}
```

### 开启 http 服务，执行 buildsStart 钩子

```ts
async function startServer(
  server: ViteDevServer,
  inlinePort?: number,
  isRestart: boolean = false
): Promise<void> {
  const httpServer = server.httpServer;
  if (!httpServer) {
    throw new Error("Cannot call server.listen in middleware mode.");
  }

  const options = server.config.server;
  // 端口
  const port = inlinePort ?? options.port ?? DEFAULT_DEV_PORT;
  // host设置
  const hostname = await resolveHostname(options.host);
  // 协议
  const protocol = options.https ? "https" : "http";
  const serverPort = await httpServerStart(httpServer, {
    port,
    strictPort: options.strictPort,
    host: hostname.host,
    logger: server.config.logger,
  });
  // 是否自动打开浏览器
  if (options.open && !isRestart) {
    const path =
      typeof options.open === "string" ? options.open : server.config.base;
    openBrowser(
      path.startsWith("http")
        ? path
        : new URL(path, `${protocol}://${hostname.name}:${serverPort}`).href,
      true,
      server.config.logger
    );
  }
}
// 初始化启动http服务，执行buildsStart钩子
export async function httpServerStart(
  httpServer: HttpServer,
  serverOptions: {
    port: number;
    strictPort: boolean | undefined;
    host: string | undefined;
    logger: Logger;
  }
): Promise<number> {
  let { port, strictPort, host, logger } = serverOptions;

  return new Promise((resolve, reject) => {
    const onError = (e: Error & { code?: string }) => {
      if (e.code === "EADDRINUSE") {
        if (strictPort) {
          httpServer.removeListener("error", onError);
          reject(new Error(`Port ${port} is already in use`));
        } else {
          logger.info(`Port ${port} is in use, trying another one...`);
          httpServer.listen(++port, host);
        }
      } else {
        httpServer.removeListener("error", onError);
        reject(e);
      }
    };

    httpServer.on("error", onError);
    // 监听端口
    httpServer.listen(port, host, () => {
      httpServer.removeListener("error", onError);
      resolve(port);
    });
  });
}
```

### 其他函数摘要

```ts
// 从配置文件中加载配置
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

## vite build

调用 `Rollup.js` 打包，详情参见 [Rollup.js 源码摘要](/vite/rollup-source.html#打包核心函数)。

```ts
const { build } = await import("./build");

try {
  await build({
    root,
    base: options.base,
    mode: options.mode,
    configFile: options.config,
    logLevel: options.logLevel,
    clearScreen: options.clearScreen,
    optimizeDeps: { force: options.force },
    build: buildOptions,
  });
} catch (e) {
  createLogger(options.logLevel).error(
    colors.red(`error during build:\n${e.stack}`),
    { error: e }
  );
  process.exit(1);
} finally {
  stopProfiler((message) => createLogger(options.logLevel).info(message));
}
// Bundles the app for production.
// Returns a Promise containing the build result.
export async function build(
  inlineConfig: InlineConfig = {}
): Promise<RollupOutput | RollupOutput[] | RollupWatcher> {
  // 解析、获取配置
  const config = await resolveConfig(
    inlineConfig,
    "build",
    "production",
    "production"
  );
  // 调用rollup打包核心函数
  let bundle: RollupBuild | undefined
  const { rollup } = await import("rollup");
  bundle = await rollup(rollupOptions);
  // 返回打包产物
  const res = [];
  for (const output of normalizedOutputs) {
    res.push(await bundle[options.write ? "write" : "generate"](output));
  }
  return Array.isArray(outputs) ? res : res[0];
} catch (e) {
  outputBuildError(e)
  throw e
} finally {
  if (bundle) await bundle.close()
}
```

[Rollup 打包核心函数](/vite/rollup-source.html#打包核心函数)

## PS：第三方库名录
- 命令行参数处理: [cac](https://github.com/cacjs/cac)
- 命令行样式(颜色): [picocolors](https://github.com/alexeyraspopov/picocolors)
- 打包器(认准ESM): [esbuild](https://github.com/evanw/esbuild) 
- Node.js中间件层(插件层): [connect](https://github.com/senchalabs/connect) 
- 跨平台文件watching库: [chokidar](https://github.com/paulmillr/chokidar) 
