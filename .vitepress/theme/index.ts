import { VPTheme } from "@vue/theme";
import { h } from "vue";
import NavbarTitle from "./components/NavbarTitle.vue";
import Poetry from "./components/Poetry.vue";
import SpeechButton from "./components/SpeechButton.vue";

import "./override.css";

export default {
  ...VPTheme,
  Layout() {
    // @ts-ignore
    return h(VPTheme.Layout, null, {
      "navbar-title": () => h(NavbarTitle),
      "aside-mid": () => h(Poetry),
    });
  },
  enhanceApp({ app }: { app: any }) {
    app.component("SpeechButton", SpeechButton);
  }
};
