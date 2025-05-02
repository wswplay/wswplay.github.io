<template>
  <section id="hero">
    <h1 class="tagline">
      {{ homeInfo.action }}
      <span>{{ homeInfo.tool }}</span>
      <br />
      <span class="accent">{{ homeInfo.target }}</span>
    </h1>
    <p class="description">{{ homeInfo.desc }}</p>
    <a id="vuemastery-action" :href="homeInfo.goto">
      {{ homeInfo.mainBtnText }}
      <svg
        class="icon-play"
        aria-labelledby="simpleicons-play-icon"
        role="img"
        viewBox="0 0 100 125"
        fill="#FFFFFF"
      >
        <title id="simpleicons-play-icon" lang="en">Play icon</title>
        <path
          d="M50,3.8C24.5,3.8,3.8,24.5,3.8,50S24.5,96.2,50,96.2S96.2,75.5,96.2,50S75.5,3.8,50,3.8z M71.2,53.3l-30.8,18  c-0.6,0.4-1.3,0.5-1.9,0.5c-0.6,0-1.3-0.1-1.9-0.5c-1.2-0.6-1.9-1.9-1.9-3.3V32c0-1.4,0.8-2.7,1.9-3.3c1.2-0.6,2.7-0.6,3.8,0  l30.8,18c1.2,0.6,1.9,1.9,1.9,3.3S72.3,52.7,71.2,53.3z"
        ></path>
      </svg>
    </a>
  </section>

  <section id="highlights" class="vt-box-container">
    <div
      class="vt-box"
      v-for="{ title, brief } in homeInfo.details"
      :key="title"
    >
      <h2>{{ title }}</h2>
      <p>{{ brief }}</p>
    </div>
  </section>
</template>

<script setup lang="ts">
const homeInfo = {
  action: "Solve",
  tool: "Three Problems For",
  target: "AI",
  desc: "解决3个问题，给岁月以AI",
  // target: "For Civilization",
  // desc: "解决3个问题，给岁月以文明。",
  // desc: "解决3个问题，给岁月以文明。我看过许多地方的云。——边城执剑人",
  // mainBtnText: "开启「三体」Vitepress-> Vue/core -> Vite(Rollup) -> About",
  mainBtnText: "开释「三题」",
  details: [
    { title: "What", brief: "是什么？" },
    { title: "Where", brief: "哪里用到了？" },
    { title: "How", brief: "怎样用的？" },
    // { title: "What", brief: "是什么？" },
    // { title: "Where", brief: "Vue在哪里用到了？" },
    // { title: "How", brief: "Vue是怎样用的？" },
  ],
  goto: "/aiart/three-body",
  // goto: "/aiart/machine-learning/overview",
};
</script>

<style scoped>
#vuemastery-action {
  font-size: 16px;
  display: inline-block;
  border-radius: 8px;
  transition: background-color 0.5s, color 0.5s;
  position: relative;
  font-weight: 600;
  background-color: var(--vt-c-green);
  color: #fff;
  /* margin-right: 18px; */
  padding: 8px 1em 8px 3em;
}

.dark #vuemastery-action {
  color: var(--vt-c-indigo);
}

#vuemastery-action:hover {
  background-color: var(--vt-c-green-dark);
  transition-duration: 0.2s;
}

.dark #vuemastery-action:hover {
  background-color: var(--vt-c-green-light);
}

.icon-play {
  display: inline;
  position: absolute;
  left: 15px;
  top: 8px;
  margin-left: 2px;
  fill: currentColor;
  width: 1.5em;
}

section {
  padding: 42px 32px;
}

#hero {
  padding: 96px 32px;
  text-align: center;
}

.tagline {
  font-size: 76px;
  line-height: 1.25;
  font-weight: 900;
  letter-spacing: -1.5px;
  max-width: 960px;
  margin: 0px auto;
}

html:not(.dark) .accent,
.dark .tagline {
  background: -webkit-linear-gradient(315deg, #42d392 25%, #647eff);
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.description {
  max-width: 960px;
  line-height: 1.5;
  color: var(--vt-c-text-2);
  transition: color 0.5s;
  font-size: 22px;
  margin: 24px auto 40px;
}

.actions a {
  font-size: 16px;
  display: inline-block;
  background-color: var(--vt-c-bg-mute);
  padding: 8px 18px;
  font-weight: 500;
  border-radius: 8px;
  transition: background-color 0.5s, color 0.5s;
}

.actions .get-started {
  margin-right: 18px;
}

.actions .icon {
  display: inline;
  position: relative;
  top: -1px;
  margin-left: 2px;
  fill: currentColor;
  transition: transform 0.2s;
}

.actions .get-started:hover {
  transition-duration: 0.2s;
}

.actions .get-started:hover .icon {
  transform: translateX(2px);
}

.actions .get-started,
.actions .setup {
  color: var(--vt-c-text-code);
}

.actions .get-started:hover,
.actions .setup:hover {
  background-color: var(--vt-c-gray-light-4);
  transition-duration: 0.2s;
}

.dark .actions .get-started:hover,
.dark .actions .setup:hover {
  background-color: var(--vt-c-gray-dark-3);
}

#special-sponsor {
  border-top: 1px solid var(--vt-c-divider-light);
  border-bottom: 1px solid var(--vt-c-divider-light);
  padding: 12px 24px;
  display: flex;
  align-items: center;
}

#special-sponsor span {
  color: var(--vt-c-text-2);
  font-weight: 500;
  font-size: 13px;
  vertical-align: middle;
  flex: 1;
}

#special-sponsor span:first-child {
  text-align: right;
}

#special-sponsor a {
  display: flex;
  justify-content: center;
  padding: 0 24px;
}

#special-sponsor img {
  height: 42px;
  margin: -6px 0;
}

.dark #special-sponsor img {
  filter: grayscale(1) invert(1);
}

#highlights {
  max-width: 960px;
  margin: 0px auto;
  color: var(--vt-c-text-2);
}

#highlights h2 {
  font-weight: 600;
  font-size: 20px;
  letter-spacing: -0.4px;
  color: var(--vt-c-green-dark);
  transition: color 0.5s;
  margin-bottom: 0.75em;
}

#highlights p {
  font-weight: 400;
  font-size: 15px;
}

#highlights .vt-box {
  background-color: transparent;
}

#sponsors {
  max-width: 900px;
  margin: 0px auto;
}

#sponsors h2 {
  font-size: 20px;
  font-weight: 600;
  margin-bottom: 1em;
}

#sponsors .sponsor-container {
  margin-bottom: 3em;
}

@media (max-width: 960px) {
  .tagline {
    font-size: 64px;
    letter-spacing: -0.5px;
  }
  .description {
    font-size: 18px;
    margin-bottom: 48px;
  }
}

@media (max-width: 768px) {
  .tagline {
    font-size: 48px;
    letter-spacing: -0.5px;
  }
}

@media (max-width: 576px) {
  #hero {
    padding: 64px 32px;
  }
  .description {
    font-size: 16px;
    margin: 18px 0 30px;
  }
  #special-sponsor {
    flex-direction: column;
  }
  #special-sponsor img {
    height: 36px;
    margin: 8px 0;
  }
  #special-sponsor span {
    text-align: center !important;
  }
  #highlights h3 {
    margin-bottom: 0.6em;
  }
  #highlights .vt-box {
    padding: 20px 36px;
  }
  .actions a {
    margin: 0.5em 0;
  }
}

@media (max-width: 370px) {
  .tagline {
    font-size: 36px;
  }
}

@media (max-width: 360px) {
  #vuemastery-action {
    margin-right: 0;
  }
}
</style>
