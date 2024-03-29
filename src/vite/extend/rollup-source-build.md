### build谱系函数集锦

```ts
runRollup(command)
  getConfigs(command)
    const configFile = await getConfigPath(command.config)
    // 加载配置文件
    const { options, warnings } = await loadConfigFile(configFile, command)
      getConfigList(fileName, commandOptions)
        getConfigFileExport(fileName, commandOptions)
          loadTranspiledConfigFile(fileName, commandOptions)
            const inputOptions = {..., plugins: [] }
            addPluginsFromCommandOption(configPlugin, inputOptions)
            // 解析配置文件
            const bundle = await rollup.rollup(inputOptions) {
              rollupInternal(rawInputOptions, null) {
                // 获取配置文件入口配置
                const { options: inputOptions } = await getInputOptions(rawInputOptions) {
                  const rawPlugins = getSortedValidatedPlugins('options')
                  const { options } = await normalizeInputOptions(
                    rawPlugins.reduce(..., rawInputOptions)) {
                    const options = {..., input: getInput(config)}
                    return { options }
                  }
                  return { options }
                }
                // 创建图谱实例
                const graph = new Graph(inputOptions, watcher) {
                  readonly modulesById = new Map()
                  this.pluginDriver = new PluginDriver(this, options, options.plugins)
                  this.moduleLoader = new ModuleLoader(this, this.modulesById)
                }
                // 进入核心构建流程
                await catchUnfinishedHookActions() {
                  try {
                    await graph.pluginDriver.hookParallel('buildStart')
                    // 构建
                    await graph.build() {
                      // 深度优先递归解析模块内容、依赖等信息，生成关系图谱
                      await this.generateModuleGraph() {
                        await this.moduleLoader.addEntryModules(
                            normalizeEntryModules(this.options.input),
                            true
                          ) {
                          const newEntryModules = await this.extendLoadModulesPromise() {
                            Promise.all(
                              unresolvedEntryModules.map(({ id, importer }) => {
                                this.loadEntryModule(id) {
                                  const resolveIdResult = await resolveId(unresolvedId) {
                                    const pluginResult = await resolveIdViaPlugins(source) {
                                      // return pluginDriver.hookFirstAndGetPlugin('resolveId')
                                    }
                                    return addJsExtensionIfNecessary(source)
                                  }
                                  return this.fetchModule(this.getResolvedIdWithDefaults(id)) {
                                    // 创建模块实例
                                    const module = new Module(id, ...)
                                    this.modulesById.set(id, module)
                                    // 添加模块源信息
                                    const loadPromise = this.addModuleSource(id, module) {
                                      try {
                                        source = await this.pluginDriver.hookFirst('load', [id])
                                      }
                                      // 更新模块相关信息
                                      module.updateOptions(sourceDescription)
                                      // 设置模块相关信息
                                      module.setSource(transform(module) {
                                        code = await pluginDriver.hookReduceArg0('transform', module.id)
                                        return { code, ... }
                                      }) {
                                        this.info.code = code
                                        const moduleAst = ast ?? this.tryParse()
                                        this.astContext = {code, ...}
                                        this.scope = new ModuleScope()
                                        this.namespace = new NamespaceVariable()
                                        this.ast = new Program()
                                        this.info.ast = moduleAst;
                                      }
                                    }.then(() => {
                                      this.getResolveStaticDependencyPromises(module),
                                      this.getResolveDynamicImportPromises(module),
                                      loadAndResolveDependenciesPromise
                                    })
                                    this.pluginDriver.hookParallel('moduleParsed')
                                    const resolveDependencyPromises = await loadPromise;
                                    await this.fetchModuleDependencies(module, ...resolveDependencyPromises)
                                    return module;
                                  }
                                }
                              }
                            ).then(entryModules => {
                              for (const [index, entryModule] of entryModules.entries()) {
                                addChunkNamesToModule(entryModule)
                                this.indexedEntryModules.push({index: xxx, module: entryModule})
                                this.indexedEntryModules.sort(({index: indexA}, {index: indexB}) => {
                                  indexA > indexB ? 1 : -1
                                })
                              }
                              return entryModules;
                            })
                          }
                          await this.awaitLoadModulesPromise()
                          return {
                            entryModules: this.indexedEntryModules.map(({ module }) => module),
                            newEntryModules
                          }
                        }
                        // 标记内部、外部模块
                        for (const module of this.modulesById.values()) {
                          if (module instanceof Module) {
                            this.modules.push(module);
                          } else {
                            this.externalModules.push(module);
                          }
                        }
                      }

                      // 按模块及依赖执行先后排序，ast按照各自类型绑定引用
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
                        this.modules = orderedModules;
                        // ast绑定引用
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

                      // 运行node.hasEffects()，标记ast属性included是否含true，决定该节点是否入包
                      this.includeStatements() {
                        for (const module of entryModules) {
                          markModuleAndImpureDependenciesAsExecuted(module) { 
                            baseModule.isExecuted = true;
                          }
                        }
                        for (const module of this.modules) {
                          if (module.info.moduleSideEffects === 'no-treeshake') {
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
                              this.includeAllExports(false)
                            } else {
                              module.include() {
                                const context = createInclusionContext()
                                if (this.ast!.shouldBeIncluded(context) {
                                  return this.included || (!context.brokenFlow && this.hasEffects(createHasEffectsContext()))
                                }) {
                                  this.ast!.include(context, false)
                                }
                              }
                            }
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
                    return handleGenerateWrite(false, ...)
                  }
                }
                return result;
              }
            }
```