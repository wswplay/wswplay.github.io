### build 函数谱系集锦

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
            const bundle = await rollup.rollup() {
              rollupInternal() {
                // 获取配置文件入口配置
                const { options: inputOptions } = await getInputOptions() {
                  getSortedValidatedPlugins('options')
                  await normalizeInputOptions()
                  return { options, unsetOptions }
                }
                // 构建图谱实例
                const graph = new Graph(inputOptions, watcher)
                // 进入核心打包流程
                await catchUnfinishedHookActions() {
                  try {
                    await graph.pluginDriver.hookParallel('buildStart')
                    // 打包
                    await graph.build() {
                      // 生成模块关系图谱
                      await this.generateModuleGraph() {
                        await this.moduleLoader.addEntryModules(normalizeEntryModules()) {
                          await this.extendLoadModulesPromise() {
                            Promise.all(
                              this.loadEntryModule() {
                                const resolveIdResult = await resolveId() {
                                  const pluginResult = await resolveIdViaPlugins() {
                                    // return pluginDriver.hookFirstAndGetPlugin('resolveId')
                                  }
                                  return addJsExtensionIfNecessary()
                                }
                                return this.fetchModule(this.getResolvedIdWithDefaults()) {
                                  const module = new Module()
                                  this.addModuleSource() {
                                    source = await this.pluginDriver.hookFirst('load')
                                    module.updateOptions(sourceDescription)
                                    module.setSource(transform(sourceDescription) {
                                      code = await pluginDriver.hookReduceArg0('transform')
                                      return { code, ... }
                                    }) {
                                      const moduleAst = ast ?? this.tryParse()
                                      this.astContext = {...}
                                      this.scope = new ModuleScope()
                                      this.namespace = new NamespaceVariable()
                                      this.ast = new Program()
                                      this.info.ast = moduleAst;
                                    }
                                  }
                                  this.pluginDriver.hookParallel('moduleParsed')
                                  await this.fetchModuleDependencies(module)
                                  return module;
                                }
                              }
                            ).then(entryModules => {
                              return entryModules;
                            })
                          }
                          await this.awaitLoadModulesPromise()
                          return { entryModules, ..., newEntryModules }
                        }
                        // 标记内部、外部模块
                        if (module instanceof Module) {
                          this.modules.push(module);
                        } else {
                          this.externalModules.push(module);
                        }
                      }

                      // 排序、绑定引用
                      this.sortModules() {
                        // 排序：递归分析模块，并标记执行顺序，返回排序后的模块数组
                        const { orderedModules, cyclePaths } = analyseModuleExecution(this.entryModules) {
                          const analyseModule = (module) => {
                            if (module instanceof Module) {
                              for (const dependency of module.dependencies) {
                                analyseModule(dependency);
                              }
                            }
                            module.execIndex = nextExecIndex++;
                            analysedModules.add(module);
                          }
                          return { cyclePaths, orderedModules };
                        }
                        // 绑定引用
                        for (const module of this.modules) {
                          module.bindReferences() {
                            this.ast!.bind() {
                              for (const key of this.keys) {
                                child?.bind()
                              }
                            }
                          }
                        }
                      }

                      // 标记入包状态
                      this.includeStatements() {
                        for (const module of entryModules) {
                          markModuleAndImpureDependenciesAsExecuted(module) { 
                            baseModule.isExecuted = true;
                          }
                        }
                        for (const module of this.modules) {
                          module.includeAllInBundle() {
                            this.ast!.include(createInclusionContext(), true) {
                              // Program.ts
                              include() {
                                this.included = true
                                for (const node of this.body) {
                                  if(...) node.include(context, includeChildrenRecursively)
                                }
                              }
                            }
                            this.includeAllExports(false);
                          }
                        }
                      }
                    }
                    // 执行 buildEnd 钩子
                    await graph.pluginDriver.hookParallel('buildEnd')
                  }
                }
                const result = {
                  // 只生成。不写入
                  async generate() {
                    return handleGenerateWrite(false)
                  }
                }
                return result;
              }
            }
```