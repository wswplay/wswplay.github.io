
module.exports = {
  markdown: {
    lineNumbers: true
  },
  locales: {
    '/': {
      lang: 'en-US',
      title: 'js边城',
      description: 'js边城'
    },
    '/zh/': {
      lang: 'zh-CN',
      title: 'js边城',
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
  const [design, vscode, pageindex, basic, vue2, vue3, performance, vuecli, vuepress, vuerouter] = cateName
  return [
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
      title: vscode,
      collapsable,
      children: [
        'vscode/code-snippets',
        'vscode/short-cut-key',
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
      ]
    },
    {
      title: vue2,
      collapsable,
      children: [
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
        'vue3/renderer',
        'vue3/composition-api',
        'vue3/proxy',
      ]
    },
    {
      title: performance,
      collapsable,
      children: [
        'performance/secondly-open-first-screen',
      ]
    },
    {
      title: vuecli,
      collapsable,
      children: [
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
      title: vuepress,
      collapsable,
      children: [
        'vuepress/setting',
      ]
    },
    {
      title: vuerouter,
      collapsable,
      children: [
        'vuerouter/init',
      ]
    },
  ]
}