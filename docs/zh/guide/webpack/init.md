---
title: 初始化
---
## 灵魂问题：
1. 是**loader**先执行，还是**插件**先执行？

## 整体流程
1. 读取参数
2. 实例化 Compiler
3. entryOption 阶段，读取入口文件
4. Loader 编译对应文件，解析成 AST
5. 找到对应依赖，递归编译处理，生成 chunk
6. 输出到 dist

### shell运行webpack
即：webapck xxx --xxx --xx
```json
"version": "5.37.0",
"bin": {
  "webpack": "bin/webpack.js"
}
```
#### bin/webpack.js
```js
// bin/webpack.js
const webpack = (options, callback) => {
  const create = () => {
    let compiler;
    if (Array.isArray(options)) {
      //
    } else {
      compiler = createCompiler(options)
      -> const createCompiler = rawOptions => {
        const compiler = new Compiler(options.context)
        -> class Compiler {
            constructor() {

            }
            compile(callback) {
              this.hooks.beforeCompile.callAsync(params, err => {
                const compilation = this.newCompilation(params)
                this.hooks.make.callAsync(compilation, err => {}
              })
            },
            run(callback) {
              const run = () => {
                this.hooks.beforeRun.callAsync(this, err => {
                  if (err) return finalCallback(err);
                  this.hooks.run.callAsync(this, err => {
                    if (err) return finalCallback(err);
                    this.readRecords(err => {
                      if (err) return finalCallback(err);
                      this.compile(onCompiled);
                    });
                  });
                };
              }
              run();
            }
          }
        return compiler;
      }

    }
  }
  if (callback) {
    try {
      const { compiler, watch, watchOptions } = create();
      if (watch) {
        compiler.watch(watchOptions, callback);
      } else {
        compiler.run((err, stats) => {
          compiler.close(err2 => {
            callback(err || err2, stats);
          });
        });
      }
      return compiler;
    } catch (err) {
      process.nextTick(() => callback(err));
      return null;
    }
  } else {
    const { compiler, watch } = create();
    return compiler;
  }
}
module.exports = webpack;
```
