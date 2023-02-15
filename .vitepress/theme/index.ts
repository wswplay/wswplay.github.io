import { VPTheme } from "@vue/theme";
import { h } from "vue";
import NavbarTitle from "./components/NavbarTitle.vue";

export default {
  ...VPTheme,
  Layout() {
    return h(VPTheme.Layout, null, {
      // "sidebar-top": () => "三体",
      "navbar-title": () => h(NavbarTitle),
    });
  },
};
