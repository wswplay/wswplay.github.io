---
title: 用Vitepress重写你的博客
---

# 基于 Vite 和 Vue 的静态网页生成器

## 集成自动化部署

先看`Vitepress`官方[部署文档](https://vitepress.vuejs.org/guide/deploying#github-pages)。  
但是，它的这个文档我多次部署失败:sweat_smile:。算了。    
我用了这个：[deploy.yml](https://github.com/wswplay/wswplay.github.io/blob/main/.github/workflows/deploy.yml)

```yaml
name: V-Deploy
on:
  push:
    branches:
      - main
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-node@v3
        with:
          node-version: 16
          cache: yarn
      - run: yarn install --frozen-lockfile

      - name: Build
        run: yarn build

      - name: Deploy
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.ACCESS_TOKEN }}
          publish_dir: .vitepress/dist
```
