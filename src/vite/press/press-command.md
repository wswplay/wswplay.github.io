---
title: Vitepress命令源码分析
outline: deep
---

# vitepress 命令源码摘要

1. 调试入口文件：`src/node/cli.ts`，在目标位置打上断点。
2. 将 `VSCode终端` 调至 `JavaScript Debug` 模式，输如下命令，唤起断点调试。

```bash
pnpm run docs-dev
# "docs-dev": "wait-on -d 100 dist/node/cli.js && node ./bin/vitepress dev docs"
```

## 处理命令行入参

```ts
const argv: any = minimist(process.argv.slice(2)); // _: ['dev', 'docs']
const command = argv._[0];
const root = argv._[command ? 1 : 0];
if (root) {
  argv.root = root;
}
```

## 主流程摘要

```ts
if (!command || command === "dev") {
  const createDevServer = async () => {
    const server = await createServer(root, argv, async () => {
      await server.close();
      await createDevServer();
    });
    await server.listen();
    logVersion(server.config.logger);
    server.printUrls();
  };
  createDevServer().catch((err) => {});
} else {
  logVersion();
  if (command === "build") {
    build(root, argv).catch((err) => {
      createLogger().error(`${c.red(`build error:`)}\n${err.stack}`);
      process.exit(1);
    });
  } else if (command === "serve" || command === "preview") {
    serve(argv).catch((err) => {
      createLogger().error(
        `${c.red(`failed to start server. error:`)}\n${err.stack}`
      );
      process.exit(1);
    });
  } else if (command === "init") {
    init();
  } else {
    createLogger().error(c.red(`unknown command "${command}".`));
    process.exit(1);
  }
}
```

### createServer 创建服务

```ts
export async function createServer(
  root: string = process.cwd(),
  serverOptions: ServerOptions = {},
  recreateServer?: () => Promise<void>
) {
  const config = await resolveConfig(root) {
    // async function resolveConfig(
    //   root: string = process.cwd(),
    //   command: 'serve' | 'build' = 'serve',
    //   mode = 'development'
    // )
    // normalizePath来自vite
    root = normalizePath(path.resolve(root))
    // 解析用户配置
    // const [userConfig, configPath, configDeps] =
        await resolveUserConfig(
          root,
          command,
          mode
        ) {
          // 加载用户配置
          const configPath = supportedConfigExtensions.flatMap((ext) => [
            resolve(root, `config/index.${ext}`),
            resolve(root, `config.${ext}`)
          ]).find(fs.pathExistsSync)
          let userConfig: RawConfigExports = {}
          let configDeps: string[] = []
          if (!configPath) {
            debug(`no config file found.`)
          } else {
            // 获取配置导出信息。loadConfigFromFile来自vite
            const configExports = await loadConfigFromFile(
              { command, mode },
              configPath,
              root
            )
            // 分配配置信息和依赖
            if (configExports) {
              userConfig = configExports.config
              configDeps = configExports.dependencies.map((file) =>
                normalizePath(path.resolve(file))
              )
            }
          }
          return [await resolveConfigExtends(userConfig), configPath, configDeps]
        }
    // 解析、格式化站点配置数据
    const site = await resolveSiteData(root, userConfig) {
      userConfig = userConfig || (await resolveUserConfig(root, command, mode))[0]
      return {
        lang: userConfig.lang || 'en-US',
        dir: userConfig.dir || 'ltr',
        title: userConfig.title || 'VitePress',
        titleTemplate: userConfig.titleTemplate,
        description: userConfig.description || 'A VitePress site',
        base: userConfig.base ? userConfig.base.replace(/([^/])$/, '$1/') : '/',
        head: resolveSiteDataHead(userConfig),
        appearance: userConfig.appearance ?? true,
        themeConfig: userConfig.themeConfig || {},
        locales: userConfig.locales || {},
        scrollOffset: userConfig.scrollOffset || 90,
        cleanUrls: !!userConfig.cleanUrls
      }
    }
    const srcDir = normalizePath(path.resolve(root, userConfig.srcDir || '.'))
    const outDir = ...
    const cacheDir = ...
    // 解析主题
    const userThemeDir = resolve(root, 'theme')
    const themeDir = (await fs.pathExists(userThemeDir)) ? userThemeDir : DEFAULT_THEME_PATH
    // 解析页面、路由
    const { pages, dynamicRoutes, rewrites } = await resolvePages(srcDir, userConfig) {
      const allMarkdownFiles = (await fg(['**.md'], {
          cwd: srcDir,
          ignore: ['**/node_modules', ...(userConfig.srcExclude || [])]
        })
      ).sort()
      const pages: string[] = []
      const dynamicRouteFiles: string[] = []
      allMarkdownFiles.forEach((file) => {
        dynamicRouteRE.lastIndex = 0
        ;(dynamicRouteRE.test(file) ? dynamicRouteFiles : pages).push(file)
      })
      const dynamicRoutes = await resolveDynamicRoutes(srcDir, dynamicRouteFiles)
      pages.push(...dynamicRoutes.routes.map((r) => r.path))
      const rewrites = resolveRewrites(pages, userConfig.rewrites)
      return { pages, dynamicRoutes, rewrites }
    }
    const config: SiteConfig = {
      root,
      srcDir,
      site,
      themeDir,
      pages,
      dynamicRoutes,
      ...
    }
    global.VITEPRESS_CONFIG = config
    return config
  }

  if (serverOptions.base) {
    config.site.base = serverOptions.base;
    delete serverOptions.base;
  }
  dns.setDefaultResultOrder("verbatim");

  return createViteServer({
    root: config.srcDir,
    base: config.site.base,
    cacheDir: config.cacheDir,
    plugins: await createVitePressPlugin(config, false, {}, {}, recreateServer),
    server: serverOptions,
    customLogger: config.logger,
  });
}
```

## 辅助信息集锦

```ts
async function resolveConfigExtends(
  config: RawConfigExports
): Promise<UserConfig> {
  const resolved = await (typeof config === "function" ? config() : config);
  if (resolved.extends) {
    const base = await resolveConfigExtends(resolved.extends);
    return mergeConfig(base, resolved);
  }
  return resolved;
}
```
