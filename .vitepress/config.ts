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
    {
      text: "Vue",
      activeMatch: "^/(vue3|vue2)/",
      items: [
        { text: "Vue3", link: "/vue3/" },
        { text: "Vue2", link: "/vue2/" },
      ],
    },
    {
      text: "Vite",
      activeMatch: "^/vite/",
      link: "/vite/",
    },
    {
      text: "Vitepress",
      activeMatch: "^/press/",
      link: "/press/",
    },
  ];
}
function geneSidebar(): ThemeConfig["sidebar"] {
  return {
    "/vue3/": [
      {
        text: "Vue3的设计与实现",
        items: [
          { text: "Proxy", link: "/vue3/proxy" },
          { text: "Reflect", link: "/vue3/reflect" },
          { text: "Ref", link: "/vue3/ref" },
        ],
      },
    ],
    "/vue2": [
      {
        text: "Vue2的原理",
        items: [{ text: "defineProperty", link: "/vue2/defineproperty" }],
      },
    ],
    "/vite/": [
      {
        text: "Vite的原理",
        items: [{ text: "Rollup", link: "/vite/rollup" }],
      },
    ],
  };
}
