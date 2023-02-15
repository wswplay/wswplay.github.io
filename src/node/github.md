---
title: Github Token设置及Pages自动部署
---

# Github：号称全球最大的交友网站

## 设置 token

有时，Github 提交，需要输入用户名、密码。但密码是不行的，会显示校验失效，它要求你用 token，于是就只能生成 token，然后贴到密码的位置咯。

```bash
头像 -> Settings -> Developer settings
-> Personal access tokens -> Generate new token -> repo
```

## 集成自动化部署

#### Travis CI

网上很多教程都是垃圾！  
唯有这个很好、很高效——[将 Hexo 部署到 GitHub Pages](https://hexo.io/zh-cn/docs/github-pages)
:::tip
1、将 [Travis CI](https://github.com/marketplace/travis-ci) 添加到你的 GitHub 账户中。  
2、前往 GitHub 的 [Applications settings](https://github.com/settings/installations)，配置 Travis CI 权限，使其能够访问你的 repository。  
3、你应该会被重定向到 Travis CI 的页面。如果没有，请 [手动前往](https://www.travis-ci.com/)。  
4、在浏览器内新建一个标签页，前往 GitHub 新建 [Personal Access Token](https://github.com/settings/tokens)，只勾选 repo 的权限并生成一个新的 Token。Token 生成后请复制并保存好。  
5、回到 Travis CI，前往你的 repository 的设置页面，在 Environment Variables 下新建一个环境变量，Name 为 **dep_wswplay**，Value 为刚才你在 GitHub 生成的 Token。确保 DISPLAY VALUE IN BUILD LOG 保持 不被勾选 避免你的 Token 泄漏。点击 Add 保存。  
6、在你的 Hexo 站点文件夹中新建一个 .travis.yml 文件：

```bash
language: node_js
node_js:
  - lts/*
install:
  - yarn install
script:
  - yarn build
deploy:
  provider: pages
  skip_cleanup: true
  local_dir: docs/.vuepress/dist
  github_token: $dep_wswplay # $为变量符号。注意变量名应与第5步Name一致。
  keep_history: true
  on:
    branch: main
```

附存档：[我的 Travis.ci 地址](https://app.travis-ci.com/github/wswplay/wswplay.github.io)
:::
