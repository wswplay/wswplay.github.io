import{_ as s,c as a,o as l,a as n}from"./app.660eab4d.js";const d=JSON.parse('{"title":"Nodejs-url模块及扩展介绍与使用","description":"","frontmatter":{"title":"Nodejs-url模块及扩展介绍与使用"},"headers":[{"level":2,"title":"url 模块","slug":"url-模块","link":"#url-模块","children":[{"level":3,"title":"url.fileURLToPath(url)","slug":"url-fileurltopath-url","link":"#url-fileurltopath-url","children":[]}]}],"relativePath":"node/core/url.md"}'),e={name:"node/core/url.md"},o=n(`<h1 id="url-模块及扩展" tabindex="-1">url 模块及扩展 <a class="header-anchor" href="#url-模块及扩展" aria-hidden="true">#</a></h1><h2 id="url-模块" tabindex="-1">url 模块 <a class="header-anchor" href="#url-模块" aria-hidden="true">#</a></h2><h3 id="url-fileurltopath-url" tabindex="-1">url.fileURLToPath(url) <a class="header-anchor" href="#url-fileurltopath-url" aria-hidden="true">#</a></h3><p>此函数可确保正确解码百分比编码字符，并确保跨平台有效的绝对路径字符串。</p><div class="language-js line-numbers-mode"><button title="Copy Code" class="copy"></button><span class="lang">js</span><pre class="shiki material-theme-palenight" tabindex="0"><code><span class="line"><span style="color:#89DDFF;font-style:italic;">import</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">{</span><span style="color:#F07178;"> </span><span style="color:#A6ACCD;">fileURLToPath</span><span style="color:#F07178;"> </span><span style="color:#89DDFF;">}</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;font-style:italic;">from</span><span style="color:#A6ACCD;"> </span><span style="color:#89DDFF;">&quot;</span><span style="color:#C3E88D;">node:url</span><span style="color:#89DDFF;">&quot;</span><span style="color:#89DDFF;">;</span></span>
<span class="line"></span>
<span class="line"><span style="color:#C792EA;">const</span><span style="color:#A6ACCD;"> __filename </span><span style="color:#89DDFF;">=</span><span style="color:#A6ACCD;"> </span><span style="color:#82AAFF;">fileURLToPath</span><span style="color:#A6ACCD;">(</span><span style="color:#89DDFF;font-style:italic;">import</span><span style="color:#89DDFF;">.</span><span style="color:#A6ACCD;">meta</span><span style="color:#89DDFF;">.</span><span style="color:#A6ACCD;">url)</span><span style="color:#89DDFF;">;</span></span>
<span class="line"></span>
<span class="line"><span style="color:#89DDFF;">new</span><span style="color:#A6ACCD;"> </span><span style="color:#82AAFF;">URL</span><span style="color:#A6ACCD;">(</span><span style="color:#89DDFF;">&quot;</span><span style="color:#C3E88D;">file:///C:/path/</span><span style="color:#89DDFF;">&quot;</span><span style="color:#A6ACCD;">)</span><span style="color:#89DDFF;">.</span><span style="color:#A6ACCD;">pathname</span><span style="color:#89DDFF;">;</span><span style="color:#A6ACCD;"> </span><span style="color:#676E95;font-style:italic;">// 错误: /C:/path/</span></span>
<span class="line"><span style="color:#82AAFF;">fileURLToPath</span><span style="color:#A6ACCD;">(</span><span style="color:#89DDFF;">&quot;</span><span style="color:#C3E88D;">file:///C:/path/</span><span style="color:#89DDFF;">&quot;</span><span style="color:#A6ACCD;">)</span><span style="color:#89DDFF;">;</span><span style="color:#A6ACCD;"> </span><span style="color:#676E95;font-style:italic;">// 正确: C:\\path\\ (Windows)</span></span>
<span class="line"></span></code></pre><div class="line-numbers-wrapper" aria-hidden="true"><span class="line-number">1</span><br><span class="line-number">2</span><br><span class="line-number">3</span><br><span class="line-number">4</span><br><span class="line-number">5</span><br><span class="line-number">6</span><br></div></div>`,5),p=[o];function t(r,c,i,D,u,y){return l(),a("div",null,p)}const C=s(e,[["render",t]]);export{d as __pageData,C as default};