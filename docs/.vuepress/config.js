module.exports = {
  locales: {
    '/': {
      lang: 'en-US',
      title: 'js边城',
      description: 'js边城'
    },
    '/zh/': {
      lang: 'zh-CN',
      title: 'js边城',
      description: '',
    }
  },
  head: [
    ['link', { rel: 'icon', href: 'https://avatars.githubusercontent.com/u/13958395?s=460&u=b30a9731e3748ced50e5b17402ab59f15f59ae05&v=4' }]
  ],
  themeConfig: {
    repo: 'wswplay/wswplay.github.io',
    editLinks: true,
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
          '/zh/guide/': getGuideSidebar('扉页', '基础知识', 'Vue2.x')
        }
      }
    }
  },
  plugins: []
}
// tools
function getGuideSidebar (groupA, groupB, groupC) {
  return [
    {
      title: groupA,
      collapsable: false,
      children: [
        '',
      ]
    },
    {
      title: groupB,
      collapsable: false,
      children: [
        'basic/memory-stack',
        'basic/process-thread',
        'basic/tcp-http',
        'basic/data-type',
      ]
    },
    {
      title: groupC,
      collapsable: false,
      children: [
        'vue2/define-react',
        'vue2/componentization',
        'vue2/compile',
      ]
    }
  ]
}