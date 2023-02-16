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
      text: "Vue",
      activeMatch: "^/vue/",
      link: "/vue/vue3/",
    },
    {
      text: "Vite",
      activeMatch: "^/vite/",
      link: "/vite/",
    },
    {
      text: "Node",
      activeMatch: "^/node/",
      link: "/node/",
    },
    {
      text: "JS基础",
      activeMatch: "^/core/",
      link: "/core/",
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
        items: [
          { text: "DefineProperty", link: "/vue/vue2/defineProperty" },
        ],
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
          { text: "Vite", link: "/vite/" },
          { text: "Rollup", link: "/vite/rollup" },
        ],
      },
    ],
    "/node/": [
      {
        text: "Node系",
        items: [
          { text: "简介", link: "/node/" },
          { text: "path方法", link: "/node/path" },
        ],
      },
      {
        text: "周边工具",
        items: [
          { text: "npm", link: "/node/npm" },
          { text: "package.json", link: "/node/package-json" },
          { text: "nvm", link: "/node/nvm" },
          { text: "git", link: "/node/git" },
          { text: "Github", link: "/node/github" },
        ],
      },
    ],
    "/core/": [
      {
        text: "JavaScript基础",
        items: [
          { text: "简介", link: "/core/" },
        ],
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
