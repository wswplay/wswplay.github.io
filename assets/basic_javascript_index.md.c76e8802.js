import{_ as e,c as a,o,a as s}from"./app.c27daf68.js";const D=JSON.parse('{"title":"JavaScript基础","description":"","frontmatter":{"title":"JavaScript基础"},"headers":[{"level":2,"title":"模块化及规范","slug":"模块化及规范","link":"#模块化及规范","children":[]},{"level":2,"title":"浏览器和 node 中使用 esm","slug":"浏览器和-node-中使用-esm","link":"#浏览器和-node-中使用-esm","children":[{"level":3,"title":"浏览器(客户端)","slug":"浏览器-客户端","link":"#浏览器-客户端","children":[]},{"level":3,"title":"node(服务端)","slug":"node-服务端","link":"#node-服务端","children":[]}]}],"relativePath":"basic/javascript/index.md"}'),n={name:"basic/javascript/index.md"},r=s(`<h1 id="javascript-基础" tabindex="-1">JavaScript 基础 <a class="header-anchor" href="#javascript-基础" aria-hidden="true">#</a></h1><h2 id="模块化及规范" tabindex="-1">模块化及规范 <a class="header-anchor" href="#模块化及规范" aria-hidden="true">#</a></h2><p>模块化，就是复杂程序按照规范拆分成相互独立的文件，同时对外暴露一些数据或方法与外部整合。模块化主要特点是：<strong>可复用性、可组合性、独立性、中心化</strong></p><blockquote><p>解决了哪些问题？<br><code>解决了命名冲突</code>：因为每个模块是独立的，所以变量或函数名重名不会发生冲突<br><code>提高可维护性</code>：因为每个文件的职责单一，有利于代码维护<br><code>性能优化</code>：异步加载模块对页面性能会非常好<br><code>模块的版本管理</code>：通过别名等配置，配合构建工具，可以实现模块的版本管理<br><code>跨环境共享模块</code>：通过 Sea.js 的 NodeJS 版本，可以实现模块的跨服务器和浏览器共享</p></blockquote><p><strong>主流标准有</strong>：CommonJS、AMD、CMD、UMD、ES6 【<a href="https://juejin.cn/post/6996595779037036580#heading-0" target="_blank" rel="noreferrer">参考资料</a>】</p><h4 id="commonjs-cjs" tabindex="-1">CommonJS(cjs) <a class="header-anchor" href="#commonjs-cjs" aria-hidden="true">#</a></h4><p>Node 用的就是 CommonJS 模块化规范。</p><h4 id="amd" tabindex="-1">AMD <a class="header-anchor" href="#amd" aria-hidden="true">#</a></h4><p>CommonJS 规范加载模块是同步加载，只有加载完成，才能执行后面的操作，而 AMD 是异步加载模块，可以指定回调函数。该规范的实现就是 require.js。</p><h4 id="cmd" tabindex="-1">CMD <a class="header-anchor" href="#cmd" aria-hidden="true">#</a></h4><p>CMD 规范整合了上面说的 CommonJS 规范和 AMD 规范的特点，CMD 规范的实现就是 sea.js。CMD 规范最大的特点就是<strong>懒加载</strong>，并且同时支持同步和异步加载模块。</p><h4 id="umd" tabindex="-1">UMD <a class="header-anchor" href="#umd" aria-hidden="true">#</a></h4><p>UMD 没有专门的规范，而是集合了上面说的三个规范于一身，它可以让我们在合适的环境选择合适的模块规范。<br> 比如在 Node.js 环境中用 CommonJS 模块规范管理，在浏览器端支持 AMD 的话就采用 AMD 模块规范，不支持就导出为全局函数。</p><h4 id="es6-模块化-esm" tabindex="-1">ES6 模块化(esm) <a class="header-anchor" href="#es6-模块化-esm" aria-hidden="true">#</a></h4><p>CommonJS 和 AMD 都是在运行时确定依赖关系，也就是运行时加载，CommonJS 加载的是拷贝，而 ES6 module 是在编译时就确定依赖关系，所有的加载都是引用，这样做的好处是可以执行静态分析和类型检查。</p><div class="tip custom-block"><p class="custom-block-title">ES6 Module 和 CommonJS 的区别：</p><ul><li>ES6 Module 的 import 是静态引入，CommonJS 的 require 是动态引入</li><li>Tree-Shaking 就是通过 ES6 Module 的 import 来进行静态分析，并且只支持 ES6 Module 模块的使用。Tree-Shaking 就是移除掉 JS 上下文中没有引用的代码，比如 import 导入模块没有返回值的情况下，webpack 在打包编译时 Tree-Shaking 会默认忽略掉此文件</li><li>ES6 Module 是对模块的引用，输出的是值的引用，改变原来模块中的值引用的值也会改变；CommonJS 是对模块的拷贝，修改原来模块的值不会影响引用的值</li><li>ES6 Module 里的 this 指向 undefined；CommonJS 里的 this 指向模块本身</li><li>ES6 Module 是在编译时确定依赖关系，生成接口并对外输出；CommonJS 是在运行时加载模块</li><li>ES6 Module 可以单独加载某个方法；CommonJS 是加载整个模块</li><li>ES6 Module 不能被重新赋值，会报错；CommonJS 可以重新赋值(改变 this 指向)</li></ul></div><h2 id="浏览器和-node-中使用-esm" tabindex="-1">浏览器和 node 中使用 esm <a class="header-anchor" href="#浏览器和-node-中使用-esm" aria-hidden="true">#</a></h2><h3 id="浏览器-客户端" tabindex="-1">浏览器(客户端) <a class="header-anchor" href="#浏览器-客户端" aria-hidden="true">#</a></h3><p>添加 <code>type=&quot;module&quot;</code> 标识。</p><div class="language-html line-numbers-mode"><button title="Copy Code" class="copy"></button><span class="lang">html</span><pre class="shiki material-theme-palenight" tabindex="0"><code><span class="line"><span style="color:#89DDFF;">&lt;</span><span style="color:#F07178;">script</span><span style="color:#89DDFF;"> </span><span style="color:#C792EA;">type</span><span style="color:#89DDFF;">=</span><span style="color:#89DDFF;">&quot;</span><span style="color:#C3E88D;">module</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;"> </span><span style="color:#C792EA;">src</span><span style="color:#89DDFF;">=</span><span style="color:#89DDFF;">&quot;</span><span style="color:#C3E88D;">./xiao.js</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">&gt;&lt;/</span><span style="color:#F07178;">script</span><span style="color:#89DDFF;">&gt;</span></span>
<span class="line"></span></code></pre><div class="line-numbers-wrapper" aria-hidden="true"><span class="line-number">1</span><br></div></div><h3 id="node-服务端" tabindex="-1">node(服务端) <a class="header-anchor" href="#node-服务端" aria-hidden="true">#</a></h3><p>文件后缀改成 <code>.mjs</code> 即可。</p>`,22),l=[r];function t(d,i,c,p,h,m){return o(),a("div",null,l)}const S=e(n,[["render",t]]);export{D as __pageData,S as default};