---
title: Vue3 Vitepress官方文档主题介绍及使用
outline: deep
---

# Vue3 官方文档 theme 介绍

此主题源于/用于`Vue3`官方文档，本站也是。
由于主题本身没有说明文档，且有些地方是硬编码，所以不太好直接用。只能看源码，去一点点的照着来。

【[Github 库地址](https://github.com/vuejs/theme)】  
【[css 基础颜色变量地址](https://github.com/vuejs/theme/blob/0496c884e37cf52a3c5775aec8d57bdd4c8e20af/src/core/styles/variables.css)】

## 站点配置

站点配置文件：`.vitepress/config.ts`。  
用非默认主题时，`Vitepress`建议用`defineConfigWithTheme`方法，提供类型推导。

```ts
import { defineConfigWithTheme } from "vitepress";
import { type Config as ThemeConfig } from "@vue/theme";
import baseConfig from "@vue/theme/config";

export default defineConfigWithTheme<ThemeConfig>({
  extends: baseConfig, // baseConfig必须要有
  ...
});
```

### 类型定义

```ts
declare function defineConfigWithTheme<ThemeConfig>(
  config: UserConfig<ThemeConfig>
): UserConfig<ThemeConfig>;

interface UserConfig<ThemeConfig = any> {
  extends?: RawConfigExports<ThemeConfig>;
  base?: string;
  lang?: string;
  title?: string;
  titleTemplate?: string | boolean;
  description?: string;
  head?: HeadConfig[];
  appearance?: boolean | "dark";
  themeConfig?: ThemeConfig;
  locales?: Record<string, LocaleConfig>;
  markdown?: MarkdownOptions;
  lastUpdated?: boolean;
  vue?: Options; // Options to pass on to `@vitejs/plugin-vue`
  vite?: UserConfig$1; // Vite config
  srcDir?: string;
  srcExclude?: string[];
  outDir?: string;
  shouldPreload?: (link: string, page: string) => boolean;
  ...
}
```

## 主题配置

主题配置文件：`.vitepress/theme/index.ts`。

### 启用主题和各种 slot

```ts
import { VPTheme } from "@vue/theme";
import { h } from "vue";
import NavbarTitle from "./components/NavbarTitle.vue";

export default {
  ...VPTheme,
  Layout() {
    return h(VPTheme.Layout, null, {
      "navbar-title": () => h(NavbarTitle),
      // 还有下面这些slots可用
      "sidebar-top": () => h("div", "hello top"),
      "sidebar-bottom": () => h("div", "hello bottom"),
      "content-top": () => h("h1", "Announcement!"),
      "content-bottom": () => h("div", "Some ads"),
      "aside-top": () => h("div", "this could be huge"),
      "aside-mid": () => h("div", { style: { height: "300px" } }, "Sponsors"),
      "aside-bottom": () =>
        h("div", { style: { height: "300px" } }, "Sponsors"),
    });
  },
};
```

### 注册全局组件/数据

```ts
import { VPTheme } from "@vue/theme";
import { h, App } from "vue";
import VueSchoolLink from './components/VueSchoolLink.vue'

export default {
  ...VPTheme,
  Layout() {...},
  enhanceApp({ app }: { app: App }) {
    app.provide('prefer-composition', "全局数据")
    app.component('VueSchoolLink', VueSchoolLink)
  }
};
```

### 相关类型定义

```ts
export type Awaitable<T> = T | PromiseLike<T>;
interface EnhanceAppContext {
  app: App;
  router: Router;
  siteData: Ref<SiteData>;
}
interface Theme {
  Layout: Component;
  NotFound?: Component;
  enhanceApp?: (ctx: EnhanceAppContext) => Awaitable<void>;
  setup?: () => void;
}
import { Theme } from "vitepress";
const VPTheme: Theme = {
  Layout: withConfigProvider(VPApp),
  NotFound: VPNotFound,
};
```
