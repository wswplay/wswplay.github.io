---
title: Vue3 Vitepress官方文档主题介绍及使用
---

# Vue3 官方文档 theme 介绍

此主题源于/用于`Vue3`官方文档，本站也是。[Github 地址](https://github.com/vuejs/theme)     
由于主题本身没有说明文档，且有些地方是硬编码，所以不太好直接用。只能看源码，去一点点的照着来。

## .vitepress/config.ts

用非默认主题时，`Vitepress`建议用`defineConfigWithTheme`方法定义一下设置文件。

```ts
import { defineConfigWithTheme } from "vitepress";
import { type Config as ThemeConfig } from "@vue/theme";
import baseConfig from "@vue/theme/config";

export default defineConfigWithTheme<ThemeConfig>({
  extends: baseConfig, // baseConfig必须要有
  ...
});
```

## .vitepress/theme/index.ts

```ts
import { VPTheme } from "@vue/theme";
import { h } from "vue";
import NavbarTitle from "./components/NavbarTitle.vue";

export default {
  ...VPTheme,
  Layout() {
    return h(VPTheme.Layout, null, {
      "navbar-title": () => h(NavbarTitle),
    });
  },
};
```

:::tip slot 说明
`navbar-title`：可定义网站`logo`和网站名。位置在左上角，即顶部导航的最左边。
:::

#### VPTheme 和 theme 类型定义

```ts
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
