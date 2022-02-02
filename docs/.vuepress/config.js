
module.exports = {
  markdown: {
    lineNumbers: true
  },
  locales: {
    '/': {
      lang: 'en-US',
      title: 'JavaScript边城',
      description: 'JavaScript边城'
    },
    '/zh/': {
      lang: 'zh-CN',
      title: 'JavaScript边城',
      description: 'JavaScript基础知识,Vue源码解读等。',
    }
  },
  head: [
    ['link', { rel: 'icon', href: 'https://avatars.githubusercontent.com/u/13958395?s=460&u=b30a9731e3748ced50e5b17402ab59f15f59ae05&v=4' }]
  ],
  themeConfig: {
    repo: 'wswplay/wswplay.github.io',
    // editLinks: true,
    docsDir: 'docs',
    logo: 'https://avatars.githubusercontent.com/u/13958395?s=460&u=b30a9731e3748ced50e5b17402ab59f15f59ae05&v=4',
    locales: {
      '/': {
        label: 'English',
        selectText: 'Languages',
        ariaLabel: 'Select language',
        editLinkText: 'Edit this page on GitHub',
        // lastUpdated: 'Last Updated',
        nav: require('./nav/en'),
      },
      '/zh/': {
        label: '简体中文',
        selectText: '选择语言',
        ariaLabel: '选择语言',
        editLinkText: '在 GitHub 上编辑此页',
        // lastUpdated: '上次更新',
        nav: require('./nav/zh'),
        sidebar: {
          '/zh/guide/': getGuideSidebar(...require('./cate'))
        }
      }
    }
  },
  plugins: []
}
// tools
function getGuideSidebar (...cateName) {
  const collapsable = true
  const [typescript, question, avr, three, design, vscode, pageindex, basic, technics, vue2, vue3, vuerouter, vuex, performance, node, vuecli, vuepress, webpack, vite, git, linux, mac, tools] = cateName
  return [
    {
      title: typescript,
      collapsable,
      children: [
        'typescript/init',
      ]
    },
    {
      title: question,
      collapsable,
      children: [
        'question/bank',
      ]
    },
    {
      title: technics,
      collapsable,
      children: [
        'technics/download',
        'technics/transdata',
        'technics/component',
      ]
    },
    {
      title: design,
      collapsable,
      children: [
        'design/essence-of-components',
        'design/create-element-fun',
        'design/render',
      ]
    },
    {
      title: performance,
      collapsable,
      children: [
        'performance/secondly-open-first-screen',
        'performance/compile',
      ]
    },
    {
      title: vue2,
      collapsable,
      children: [
        'vue2/vision',
        'vue2/data-driven',
        'vue2/componentization',
        'vue2/component-recursion',
        'vue2/compile',
        'vue2/diff',
        'vue2/mixin',
      ]
    },
    {
      title: vue3,
      collapsable,
      children: [
        'vue3/basic-concept',
        'vue3/flow-chart',
        'vue3/renderer',
        'vue3/composition-api',
        'vue3/reactive',
        'vue3/ref',
        'vue3/computed',
        'vue3/watch',
        'vue3/life-cycle',
        'vue3/component',
        'vue3/compile',
        'vue3/directives',
        'vue3/teleport',
        'vue3/keep-alive',
        'vue3/transition',
        'vue3/router',
      ]
    },
    {
      title: basic,
      collapsable,
      children: [
        'basic/memory-stack',
        'basic/data-type',
        'basic/data-structures',
        'basic/algorithm',
        'basic/process-thread',
        'basic/tcp-http',
        'basic/module-specification',
        'basic/closure',
        'basic/recursion',
        'basic/prototype',
        'basic/operators',
        'basic/break-continue',
        'basic/promise',
        'basic/currying',
        'basic/with',
        'basic/proxy',
        'basic/reflect',
        'basic/utils',
        'basic/rest',
        'basic/iife',
        'basic/es6',
      ]
    },
    {
      title: avr,
      collapsable,
      children: [
        'avr/concept',
        'avr/three',
      ]
    },
    {
      title: three,
      collapsable,
      children: [
        'three/vertex',
        'three/material',
        'three/light',
        'three/group',
      ]
    },
    {
      title: vscode,
      collapsable,
      children: [
        'vscode/code-snippets',
        'vscode/short-cut-key',
        'vscode/search',
      ]
    },
    {
      title: pageindex,
      collapsable,
      children: [
        '',
      ]
    },
    {
      title: vuerouter,
      collapsable,
      children: [
        'vuerouter/init',
        'vuerouter/history',
      ]
    },
    {
      title: vuex,
      collapsable,
      children: [
        'vuex/init',
        'vuex/question',
      ]
    },
    {
      title: vuecli,
      collapsable,
      children: [
        'vuecli/init',
        'vuecli/cli-service',
        'vuecli/browser-compatibility',
        'vuecli/html-static-assets',
        'vuecli/css',
        'vuecli/webpack',
        'vuecli/mode-and-env',
        'vuecli/deployment',
      ]
    },
    {
      title: node,
      collapsable,
      children: [
        'node/nvm',
        'node/init',
        'node/exports',
        'node/path',
      ]
    },
    {
      title: vuepress,
      collapsable,
      children: [
        'vuepress/setting',
      ]
    },
    {
      title: webpack,
      collapsable,
      children: [
        'webpack/theory',
        'webpack/debugger',
        'webpack/init',
        'webpack/tapable',
      ]
    },
    {
      title: vite,
      collapsable,
      children: [
        'vite/init',
      ]
    },
    {
      title: git,
      collapsable,
      children: [
        'git/command',
        'git/github',
      ]
    },
    {
      title: linux,
      collapsable,
      children: [
        'linux/command',
      ]
    },
    {
      title: mac,
      collapsable,
      children: [
        'mac/short-cut-key',
      ]
    },
    {
      title: tools,
      collapsable,
      children: [
        'tools/yarn',
        'tools/monorepo-pnpm',
        'tools/python',
        'tools/npm',
        'tools/axios',
        'tools/style',
        'tools/html',
        'tools/browser',
      ]
    }
  ]
}