---
title: vite命令源码解析
outline: deep
---

# vite 命令源码解析

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
  // 全部配置
  const config = await resolveConfig(inlineConfig, "serve");
  // 获取根目录、服务器配置
  const { root, server: serverConfig } = config;
  // 声明中间件
  const middlewares = connect() as Connect.Server
  // 创建http服务对象
  const httpServer = middlewareMode
    ? null
    : await resolveHttpServer(serverConfig, middlewares, httpsOptions)
  // 开启文件wather功能
  const watcher = chokidar.watch(
    path.resolve(root),
    resolvedWatchOptions,
  ) as FSWatcher
  // 初始化模块图谱，以存储模块信息及引用关系
  const moduleGraph: ModuleGraph = new ModuleGraph((url, ssr) =>
    container.resolveId(url, undefined, { ssr }),
  )
  // 插件容器？这是干嘛用的？
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
  server.transformIndexHtml = createDevHtmlTransformFn(server)
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

### startServer

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
```

### resolveHttpServer
```ts
export async function resolveHttpServer(
  { proxy }: CommonServerOptions,
  app: Connect.Server,
  httpsOptions?: HttpsServerOptions,
): Promise<HttpServer> {
  if (!httpsOptions) {
    const { createServer } = await import('node:http')
    return createServer(app)
  }

  // #484 fallback to http1 when proxy is needed.
  if (proxy) {
    const { createServer } = await import('node:https')
    return createServer(httpsOptions, app)
  } else {
    const { createSecureServer } = await import('node:http2')
    return createSecureServer(
      {
        // Manually increase the session memory to prevent 502 ENHANCE_YOUR_CALM
        // errors on large numbers of requests
        maxSessionMemory: 1000,
        ...httpsOptions,
        allowHTTP1: true,
      },
      // @ts-expect-error TODO: is this correct?
      app,
    ) as unknown as HttpServer
  }
}
```
