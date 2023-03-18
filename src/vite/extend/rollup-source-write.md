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
                      async generate() {
                        const outputBundleBase: OutputBundle = Object.create(null)
                        const outputBundle = getOutputBundle(outputBundleBase)
                        this.pluginDriver.setOutputBundle(outputBundle, this.outputOptions)
                        try {
                          await this.pluginDriver.hookParallel('renderStart')
                          const chunks = await this.generateChunks(outputBundle, getHashPlaceholder)
                          for (const chunk of chunks) {
                            chunk.generateExports()
                          }
                          await renderChunks(...)
                          await this.pluginDriver.hookSeq('generateBundle')
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
