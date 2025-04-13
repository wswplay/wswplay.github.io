import { defineConfigWithTheme } from "vitepress";
import { type Config as ThemeConfig } from "@vue/theme";
import baseConfig from "@vue/theme/config";
import { logoUrl } from "./theme/composables/constant";

export default defineConfigWithTheme<ThemeConfig>({
  extends: baseConfig,

  // title: "JavaScript边城",
  title: "AI边城",
  head: [["link", { rel: "icon", href: logoUrl }]],
  srcDir: "src",
  themeConfig: {
    nav: geneNav(),
    sidebar: geneSidebar(),
    i18n: geneI18n(),
    socialLinks: [
      { icon: "github", link: "https://github.com/wswplay/wswplay.github.io" },
    ],
    footer: {
      copyright: `Copyright © 2020-${new Date().getFullYear()} 边城`,
    },
  },
  markdown: {
    lineNumbers: true,
    math: true,
    theme: "github-dark",
  },
});

// 工具函数
function geneNav(): ThemeConfig["nav"] {
  return [
    {
      text: "首页",
      link: "/",
    },
    {
      text: "AI·文艺",
      activeMatch: "^/aiart",
      link: "/aiart/machine-learning/overview",
    },
    // {
    //   text: "Vue",
    //   activeMatch: "^/(vue3|vue2)/",
    //   items: [
    //     { text: "Vue3", link: "/vue3/" },
    //     { text: "Vue2", link: "/vue2/" },
    //   ],
    // },
    {
      text: "Vue·周边",
      activeMatch: "^/vue/",
      link: "/vue/vue3/source-code",
    },
    {
      text: "Vite·Press",
      activeMatch: "^/vite/",
      link: "/vite/",
    },
    {
      text: "Node·周边",
      activeMatch: "^/node/",
      link: "/node/core/",
    },
    {
      text: "Js·Ts",
      activeMatch: "^/basic",
      link: "/basic/javascript/",
    },
    {
      text: "Mac·Linux",
      activeMatch: "^/mac",
      link: "/mac/setting",
    },
    {
      text: "数学·统计",
      activeMatch: "^/mathstat",
      link: "/mathstat/math/number-theory",
    },
  ];
}
function geneSidebar(): ThemeConfig["sidebar"] {
  return {
    "/vue/": [
      {
        text: "Vue3.x系列集锦",
        items: [
          { text: "Core流程源码摘要", link: "/vue/vue3/source-code" },
          { text: "Reactive响应式系统", link: "/vue/vue3/reactive" },
          { text: "Router源码摘要", link: "/vue/vue3/router-code" },
          { text: "Pinia源码摘要", link: "/vue/vue3/pinia-code" },
        ],
      },
      {
        text: "Vue2原理",
        items: [{ text: "DefineProperty", link: "/vue/vue2/defineProperty" }],
      },
    ],
    "/vite/": [
      {
        text: "Vite原理与周边",
        items: [
          { text: "Vite基础", link: "/vite/" },
          { text: "vite命令源码摘要", link: "/vite/command-cli" },
          { text: "Vite自动生成项目原理", link: "/vite/create-vite" },
          { text: "Vite插件怎么写", link: "/vite/plugin" },
        ],
      },
      {
        text: "Rollup.js",
        items: [
          { text: "安装命令及配置文件", link: "/vite/rollup" },
          { text: "rollup.js源码摘要", link: "/vite/rollup-source" },
        ],
      },
      {
        text: "Vitepress",
        items: [
          { text: "简介与功能", link: "/vite/press/vitepress" },
          { text: "vitepress命令源码摘要", link: "/vite/press/press-command" },
          { text: "默认主题(default)", link: "/vite/press/theme-default" },
          { text: "Vue官方主题(Vue3)", link: "/vite/press/theme-vue" },
        ],
      },
    ],
    "/node/": [
      {
        text: "Node基础",
        items: [
          { text: "简介", link: "/node/core/" },
          { text: "fs文件系统及扩展", link: "/node/core/fs" },
          { text: "url模块", link: "/node/core/url" },
          { text: "path路径", link: "/node/core/path" },
          { text: "http模块", link: "/node/core/http" },
          { text: "process进程", link: "/node/core/process" },
        ],
      },
      {
        text: "周边工具",
        items: [
          { text: "本地调试Node项目文件", link: "/node/external/debug" },
          { text: "写个Node命令行工具", link: "/node/external/cli" },
          { text: "npm", link: "/node/external/npm" },
          { text: "npx", link: "/node/external/npx" },
          { text: "package.json", link: "/node/external/package-json" },
          { text: "pnpm monorepo", link: "/node/external/pnpm-monorepo" },
          { text: "nvm 和 nrm", link: "/node/external/nvm" },
          { text: "git", link: "/node/external/git" },
          { text: "Github", link: "/node/external/github" },
          { text: "npm-run-all", link: "/node/external/npm-run-all" },
          { text: "服务器类型", link: "/node/external/server" },
          { text: "WebSocket", link: "/node/external/websockets" },
        ],
      },
    ],
    "/basic/": [
      {
        text: "JavaScript基础",
        items: [
          { text: "简介", link: "/basic/javascript/" },
          { text: "Array数组", link: "/basic/javascript/array" },
          { text: "Object对象", link: "/basic/javascript/object" },
          { text: "Function函数", link: "/basic/javascript/function" },
          { text: "Symbol标识符", link: "/basic/javascript/symbol" },
          { text: "Promise", link: "/basic/javascript/promise" },
          { text: "Proxy", link: "/basic/javascript/proxy" },
          { text: "Reflect", link: "/basic/javascript/reflect" },
          { text: "表达式与运算符", link: "/basic/javascript/operators" },
          { text: "位运算", link: "/basic/javascript/bitwise-operators" },
        ],
      },
      {
        text: "TypeScript基础",
        items: [
          { text: "基本认知与区别", link: "/basic/typescript/" },
          {
            text: "tsconfig.json字段详解",
            link: "/basic/typescript/config-file",
          },
          { text: "declare及声明文件", link: "/basic/typescript/declare" },
          { text: "tsx执行ts及VSCode调试ts", link: "/basic/typescript/tsx" },
        ],
      },
      {
        text: "周边及工具",
        items: [
          { text: "Markdown及生成目录", link: "/basic/tools/markdown" },
          { text: "工具函数-冷知识-酷代码", link: "/basic/tools/cold-code" },
          { text: "Web Worker", link: "/basic/tools/worker" },
        ],
      },
    ],
    "/mac/": [
      {
        text: "Mac那点事",
        items: [
          { text: "基本设置", link: "/mac/setting" },
          { text: "日常操作", link: "/mac/daily" },
          { text: "工具:brew/iTerm2", link: "/mac/dev-tools" },
          { text: "VSCode编辑器", link: "/mac/vscode" },
        ],
      },
      {
        text: "Linux",
        items: [
          { text: "基础命令", link: "/mac/linux/os-command" },
          { text: "vim编辑器", link: "/mac/linux/vim" },
        ],
      },
    ],
    "/aiart/": [
      {
        text: "机器学习",
        items: [
          { text: "概览", link: "/aiart/machine-learning/overview" },
          { text: "conda", link: "/aiart/machine-learning/conda" },
          { text: "kaggle", link: "/aiart/machine-learning/kaggle" },
        ],
      },
      {
        text: "深度学习",
        items: [
          { text: "概览", link: "/aiart/deep-learning/overview" },
          { text: "数学基础", link: "/aiart/deep-learning/mathematics" },
          { text: "基本概念", link: "/aiart/deep-learning/basic-concept" },
          { text: "线性回归", link: "/aiart/deep-learning/linear-regression" },
        ],
      },
      {
        text: "Huggingface",
        items: [
          { text: "基础配置", link: "/aiart/huggingface/config" },
          { text: "Diffusers", link: "/aiart/huggingface/diffusers" },
          { text: "Accelerate", link: "/aiart/huggingface/accelerate" },
          { text: "Pytorch", link: "/aiart/huggingface/pytorch" },
        ],
      },
      {
        text: "ComfyUI",
        items: [{ text: "基本概念", link: "/aiart/comfyui/basic" }],
      },
      {
        text: "Python",
        items: [
          { text: "安装与设置", link: "/aiart/python/basic-info" },
          { text: "pip", link: "/aiart/python/pip" },
          { text: "面向对象编程(OOP)", link: "/aiart/python/oop" },
          { text: "Pandas", link: "/aiart/python/pandas" },
          { text: "PIL", link: "/aiart/python/pil" },
        ],
      },
      {
        text: "文艺",
        items: [{ text: "孙子兵法", link: "/aiart/sunzi-war-art" }],
      },
      {
        text: "AI",
        items: [{ text: "ChatGPT问答集锦", link: "/aiart/chat-gpt" }],
      },
    ],
    "/mathstat": [
      {
        text: "理论",
        items: [
          { text: "数的概念", link: "/mathstat/math/number-theory" },
          { text: "数学符号", link: "/mathstat/math/math-symbol" },
        ],
      },
      {
        text: "微积分",
        items: [{ text: "基本定理", link: "/mathstat/math/calculus" }],
      },
      {
        text: "统计学",
        items: [],
      },
    ],
  };
}
function geneI18n(): ThemeConfig["i18n"] {
  return {
    toc: "本页目录",
    previous: "前一篇",
    next: "下一篇",
    pageNotFound: "页面未找到",
  };
}
