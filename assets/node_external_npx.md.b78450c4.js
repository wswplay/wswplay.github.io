import{_ as n,c as e,o as t,a as r}from"./app.ebd92e94.js";const _=JSON.parse('{"title":"npx是什么介绍使用作用","description":"","frontmatter":{"title":"npx是什么介绍使用作用"},"headers":[{"level":2,"title":"--no-install","slug":"no-install","link":"#no-install","children":[]},{"level":2,"title":"--ignore-existing","slug":"ignore-existing","link":"#ignore-existing","children":[]},{"level":2,"title":"-p","slug":"p","link":"#p","children":[]}],"relativePath":"node/external/npx.md"}'),a={name:"node/external/npx.md"},i=r('<h1 id="npx-npm-包执行器" tabindex="-1">npx：npm 包执行器 <a class="header-anchor" href="#npx-npm-包执行器" aria-hidden="true">#</a></h1><p>npx 是 npm5.2.0 版本新增的一个工具包，定义为 npm 包的执行者。npm 自带 npx，可以直接使用。<br><strong>相比 npm</strong>，npx 会<strong>自动安装</strong>依赖包，<strong>并执行某个命令</strong>。</p><p>npx 会在当前目录下 <code>./node_modules/.bin</code> 里去查找是否有可执行的目标命令。<br> 如果没有，再<strong>全局查找</strong>，是否有安装对应的模块。<br> 全局没有，就<strong>自动下载</strong>到一个<strong>临时目录，用完即删</strong>，不会占用本地资源。</p><h2 id="no-install" tabindex="-1">--no-install <a class="header-anchor" href="#no-install" aria-hidden="true">#</a></h2><p>告诉 npx 不要自动下载。</p><h2 id="ignore-existing" tabindex="-1">--ignore-existing <a class="header-anchor" href="#ignore-existing" aria-hidden="true">#</a></h2><p>告诉 npx 忽略本地已经存在的模块，每次都去执行下载操作，也就是每次都会下载安装临时模块并在用完后删除。</p><h2 id="p" tabindex="-1">-p <a class="header-anchor" href="#p" aria-hidden="true">#</a></h2><p>-p 用于指定 npx 所要安装的模块，它可以指定某一个版本进行安装。</p>',9),s=[i];function o(p,l,d,c,h,x){return t(),e("div",null,s)}const m=n(a,[["render",o]]);export{_ as __pageData,m as default};