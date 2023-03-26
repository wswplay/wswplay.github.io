### generate / write 函数谱系集锦

```ts
runRollup() {
  const { options, warnings } = await getConfigs(command)
  try {
    for (const inputOptions of options) {
      await build(inputOptions, warnings, command.silent) {
        const outputOptions = inputOptions.output
        // 返回 build 结果
        const bundle = await rollup(inputOptions as any) {
          return rollupInternal(rawInputOptions, null) {
            const { options: inputOptions } = await getInputOptions(rawInputOptions)
            const graph = new Graph(inputOptions, watcher)
            await catchUnfinishedHookActions(graph.pluginDriver, async () => {
              await graph.build();
            });
            const result = {
              // 写入、生成文件
              async write() {
                return handleGenerateWrite(true, inputOptions, unsetInputOptions, rawOutputOptions, graph) {
                  const { outputOptions, outputPluginDriver } = await getOutputOptionsAndPluginDriver() {
                    const outputPluginDriver = inputPluginDriver.createOutputPluginDriver(rawPlugins)
                    return { outputPluginDriver, outputOptions }
                  }
                  return catchUnfinishedHookActions() {
                    const bundle = new Bundle(outputOptions, outputPluginDriver, graph) {
                      constructor() {
                        private readonly outputOptions
                      }
                      async generate() {
                        // 声明返回数据
                        const outputBundleBase: OutputBundle = Object.create(null)
                        // 创建返回数据转发代理
                        const outputBundle = getOutputBundle(outputBundleBase)
                        this.pluginDriver.setOutputBundle(outputBundle, this.outputOptions)
                        try {
                          await this.pluginDriver.hookParallel('renderStart')
                          // 生成chunks
                          const chunks = await this.generateChunks(outputBundle) {
                            const snippets = getGenerateCodeSnippets(this.outputOptions)
                            const chunks: Chunk[] = []
                            for(const {alias, modules} of getChunkAssignments(this.graph.entryModules)) {
                              const chunk = new Chunk(modules, this.inputOptions, this.outputOptions)
                            }
                            for (const chunk of chunks) {
                              // 设置chunk依赖、导入、导出等
                              chunk.link()
                            }
                            return [...chunks, ...facades]
                          }
                          // 生成chunk导出变量、模式等
                          for (const chunk of chunks) {
                            chunk.generateExports()
                          }
                          // 渲染chunk
                          await renderChunks(chunks, outputBundle) {
                            const renderedChunks = await Promise.all(chunks.map(chunk => chunk.render()) {
                              // 预备文件名
                              const preliminaryFileName = this.getPreliminaryFileName()
                              const { xxx } = this.renderModules(preliminaryFileName.fileName) {
                                const { orderedModules } = this
                                const magicString = new MagicStringBundle({ separator: `${n}${n}` })
                                const renderOptions = {...}
                                for (const module of orderedModules) {
                                  const rendered = module.render(renderOptions)
                                }
                                return { magicString, renderedSource, ... }
                              }
                              const { intro, outro, banner, footer } = await createAddons()
                              return { chunk: this, magicString, preliminaryFileName, usedModules }
                            })
                            // 创建 chunk 图谱
                            const chunkGraph = getChunkGraph(chunks)
                            // 生成chunk哈希
                            const { nonHashedChunksWithPlaceholders } = await transformChunksAndGenerateContentHashes(renderedChunks, chunkGraph)
                            const hashesByPlaceholder = generateFinalHashes(..., outputBundle, )
                            // 整合chunk
                            addChunksToBundle(..., outputBundle, nonHashedChunksWithPlaceholders) {
                              for (const { chunk, code, fileName, map } of nonHashedChunksWithPlaceholders) {
                                outputBundle[fileName] = chunk.finalizeChunk(...)
                              }
                            }
                          }
                          // 移除未引用的物料
                          removeUnreferencedAssets(outputBundle)
                          await this.pluginDriver.hookSeq('generateBundle')
                          // 返回数据
                          return outputBundleBase;
                        }
                      }
                    }
                    const generated = await bundle.generate(isWrite)
                    if (isWrite) {
                      await Promise.all(writeOutputFile(chunk, outputOptions))
                      await outputPluginDriver.hookParallel('writeBundle')
                    }
                    return createOutput(generated) {
                      return { output: xxx }
                    }
                  }
                }
              }
            }
            return result;
          }
        }
        // 唤起写入功能
        await Promise.all(outputOptions.map(bundle.write))
        await bundle.close();
      }
    }
  }
}

```
