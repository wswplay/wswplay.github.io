---
title: 用Vitepress重写你的博客
---

# 基于 Vite 和 Vue 的静态网页生成器

## 集成自动化部署

```yaml
name: xiao-github-actions-deploy
on:
  push:
    branches:
      - main
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # 1. 安装 Node.js（官方推荐用 setup-node 管理 Node 版本）
      - uses: actions/setup-node@v4
        with:
          node-version: 22

      # 2. 用 pnpm 官方 Action 安装 pnpm（最稳的方式，自动处理 PATH 和缓存）
      - uses: pnpm/action-setup@v2
        with:
          version: latest # 默认安装最新版，也可以指定如 "8.15.0"
          run_install: false # 不自动执行 pnpm install，我们自己控制

      # 3. 安装依赖（自动走 pnpm 缓存，速度更快）
      - run: pnpm install

      # 4. 构建项目
      - run: pnpm run build

      # 5. 部署到 GitHub Pages
      - name: 部署
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: .vitepress/dist
```
