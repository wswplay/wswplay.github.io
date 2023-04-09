import { defineConfigWithTheme } from "vitepress";
import { type Config as ThemeConfig } from "@vue/theme";
import baseConfig from "@vue/theme/config";
import { logoUrl } from "./theme/composables/constant";

export default defineConfigWithTheme<ThemeConfig>({
  extends: baseConfig,

  title: "JavaScript边城",
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
  },
});

// 工具函数
function geneNav(): ThemeConfig["nav"] {
  return [
    {
      text: "首页",
      link: "/",
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
  ];
}
function geneSidebar(): ThemeConfig["sidebar"] {
  return {
    "/vue/": [
      {
        text: "Vue3.0设计与实现",
        items: [
          { text: "源码摘要", link: "/vue/vue3/source-code" },
          { text: "Ref", link: "/vue/vue3/ref" },
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
          { text: "简介与功能", link: "/vite/vitepress" },
          { text: "默认主题(default)", link: "/vite/theme-default" },
          { text: "Vue官方主题(Vue3)", link: "/vite/theme-vue" },
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
          { text: "位运算", link: "/basic/javascript/operators" },
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
        items: [{ text: "Markdown及生成目录", link: "/basic/tools/markdown" }],
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
