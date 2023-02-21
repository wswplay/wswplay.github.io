import { defineConfigWithTheme } from "vitepress";
import { type Config as ThemeConfig } from "@vue/theme";
import baseConfig from "@vue/theme/config";

export default defineConfigWithTheme<ThemeConfig>({
  extends: baseConfig,

  title: "JavaScript边城",
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
      link: "/vue/vue3/",
    },
    {
      text: "Vite·Press",
      activeMatch: "^/vite/",
      link: "/vite/",
    },
    {
      text: "Node·周边",
      activeMatch: "^/node/",
      link: "/node/",
    },
    {
      text: "Js·Ts",
      activeMatch: "^/core",
      link: "/core/javascript/",
    },
  ];
}
function geneSidebar(): ThemeConfig["sidebar"] {
  return {
    "/vue/": [
      {
        text: "Vue3设计与实现",
        items: [
          { text: "Proxy", link: "/vue/vue3/proxy" },
          { text: "Reflect", link: "/vue/vue3/reflect" },
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
        text: "Vitepress",
        items: [
          { text: "简介与功能", link: "/vite/vitepress" },
          { text: "默认主题(default)", link: "/vite/theme-default" },
          { text: "Vue官方主题(Vue3)", link: "/vite/theme-vue" },
        ],
      },
      {
        text: "Vite",
        items: [
          { text: "Vite基础", link: "/vite/" },
          { text: "Rollup", link: "/vite/rollup" },
        ],
      },
    ],
    "/node/": [
      {
        text: "Node基础",
        items: [
          { text: "简介", link: "/node/" },
          { text: "path路径", link: "/node/path" },
          { text: "process进程", link: "/node/process" },
        ],
      },
      {
        text: "周边工具",
        items: [
          { text: "本地调试Node项目文件", link: "/node/debug" },
          { text: "写个Node命令行工具", link: "/node/cli" },
          { text: "npm", link: "/node/npm" },
          { text: "npx", link: "/node/npx" },
          { text: "package.json", link: "/node/package-json" },
          { text: "pnpm和monorepo", link: "/node/pnpm-monorepo" },
          { text: "nvm", link: "/node/nvm" },
          { text: "git", link: "/node/git" },
          { text: "Github", link: "/node/github" },
          { text: "npm-run-all", link: "/node/npm-run-all" },
        ],
      },
    ],
    "/core/": [
      {
        text: "JavaScript基础",
        items: [
          { text: "简介", link: "/core/javascript/" },
          { text: "Array数组", link: "/core/javascript/array" },
        ],
      },
      {
        text: "TypeScript基础",
        items: [
          { text: "基本认知", link: "/core/typescript/" },
          { text: "declare及声明文件", link: "/core/typescript/declare" },
          { text: "用tsx直接执行ts文件", link: "/core/typescript/tsx" },
        ],
      },
      {
        text: "周边及工具",
        items: [{ text: "Markdown及生成目录", link: "/core/tools/markdown" }],
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
