### build 流程函数目录

```ts
runRollup()
  getConfigs()
    getConfigPath()
    // 加载配置文件
    loadConfigFile()
      getConfigList()
        getConfigFileExport()
          loadTranspiledConfigFile()
            addPluginsFromCommandOption()
            // 解析打包配置文件
            const bundle = await rollup.rollup()
              rollupInternal()
                // 获取配置文件入口配置
                getInputOptions()
                  getSortedValidatedPlugins("options")
                  await normalizeInputOptions()
                // 构建图谱实例
                const graph = new Graph()
                // 进入核心打包流程
                await catchUnfinishedHookActions()
                  try {
                    await graph.pluginDriver.hookParallel("buildStart")
                    // 打包
                    await graph.build() {
                      await this.generateModuleGraph() {
                        await this.moduleLoader.addEntryModules(normalizeEntryModules()) {
                          await this.extendLoadModulesPromise() {
                            Promise.all(
                              this.loadEntryModule() {
                                const resolveIdResult = await resolveId() {
                                  await resolveIdViaPlugins() {
                                    return pluginDriver.hookFirstAndGetPlugin("resolveId")
                                  }
                                  return addJsExtensionIfNecessary()
                                }
                                return this.fetchModule(this.getResolvedIdWithDefaults()) {
                                  const module = new Module()
                                  this.addModuleSource() {
                                    source = await this.pluginDriver.hookFirst('load')
                                    module.updateOptions()
                                    module.setSource(transform() {
                                      code = await pluginDriver.hookReduceArg0("transform")
                                      return { code, ast, ... }
                                    }) {
                                      const moduleAst = ast ?? this.tryParse()
                                      this.astContext = {...}
                                      this.scope = new ModuleScope()
                                      this.namespace = new NamespaceVariable()
                                      this.ast = new Program()
                                      this.info.ast = moduleAst;
                                    }
                                  }
                                  this.pluginDriver.hookParallel("moduleParsed")
                                  await this.fetchModuleDependencies()
                                  return module;
                                }
                              }
                            }).then(entryModules => {
                              return entryModules;
                            })
                          }
                          await this.awaitLoadModulesPromise()
                          return { entryModules, ..., newEntryModules }
                        }
                        if (module instanceof Module) {
                          this.modules.push(module);
                        } else {
                          this.externalModules.push(module);
                        }
                      }
                      // 排序
                      this.sortModules()
                      // 标记
                      this.includeStatements()
                    }
                    await graph.pluginDriver.hookParallel('buildEnd')
                  }
                const result = {
                  // 只生成。不写入
                  async generate() {
                    return handleGenerateWrite(false)
                  }
                }
                return result;
```