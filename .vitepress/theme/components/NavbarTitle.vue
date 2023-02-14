<template>
  <div class="nav-bar-title">
    <div class="bar-logo">
      <img :src="logoInfo.url" :alt="logoInfo.alt" />
    </div>
    <div class="title-text">{{ site.title }}</div>
  </div>
</template>

<script setup lang="ts">
import { useData } from "vitepress";
const { site } = useData();
const logoInfo = {
  url: "https://avatars.githubusercontent.com/u/13958395?v=4",
  alt: "心水喵论",
};
</script>

<style scoped lang="less">
.nav-bar-title {
  display: flex;
  justify-content: center;
  align-items: center;
  .bar-logo {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    overflow: hidden;
  }
  .title-text {
    line-height: 36px;
    font-weight: bold;
    margin: 0 5px;
  }
}
</style>
