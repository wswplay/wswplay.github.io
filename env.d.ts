declare module "@vue/theme/config" {
  import { UserConfig } from "vitepress";
  const config: () => Promise<UserConfig>;
  export default config;
}

declare module "*.vue" {
  import { DefineComponent } from "vue";
  const component: DefineComponent<{}, {}, any>;
  export default component;
}
