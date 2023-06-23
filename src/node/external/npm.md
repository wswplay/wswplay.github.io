---
title: npm命令及其使用方法
---

# Node Package Manager

## 创建`package.json`文件

```bash
# 交互式创建
npm init
# 快速创建默认项和值的文件
npm init -y
```

## 发布 npm 包

```bash
# 查看源地址
npm get registry
# 如不是npm官方源
# 则切换到官方源地址
npm config set registry https://registry.npmjs.org
# 查看是否已登录npm，如已登录则会显示用户名
npm who am i
# 如果没有登录，先登录，然后输入用户名、密码
npm login
# 发布
npm publish
# 如果包名称带有scope(如@wswplay)，则可能会发布失败
# You must sign up for private packages 意思是必须付费
# 但其实并不必须，发布为公共访问就行
npm publish --access public
# 发布成功
```

发布成功后，登录[npm 官网](https://www.npmjs.com/)个人中心`packages`中可以看到。

## 设置源地址

> 有些项目，有些时候，我们一定要切到官方的源上，才能安装。如果不，会提示一些莫名其妙的错误。

```bash
# 查看源地址
npm get registry
# 将源地址改成taobao源
npm config set registry https://registry.npm.taobao.org
# 恢复成官方源地址
npm config set registry https://registry.npmjs.org
```

## 安装指定版本

```bash
# 安装Vue3.0及以上版本
npm i vue@^3.0.0 -S
# 安装最新版本
npm i xxx@latest -D
```

## 版本号及标识

### 版本号

`npm`包版本号通用遵循`semver`语义化版本规范，版本格式为(如`1.0.0`)：`major.minor.patch`
:::tip 版本号
1、主版本号(`major`)：当你做了不兼容的 API 修改  
2、次版本号(`minor`)：当你做了向下兼容的功能性新增  
3、修订号(`patch`)：当你做了向下兼容的问题修正  
:::
先行版本号，是加到修订号的后面，作为版本号的延伸。

> 格式是在修订版本号后面加上一个连接号（-），再加上一连串以点（.）分割的标识符，标识符可以由英文、数字和连接号（[0-9A-Za-z-]）组成。

:::tip 先行号
1、`alpha`：不稳定版本，一般而言，该版本的 Bug 较多，需要继续修改，是测试版本  
2、`beta`：基本稳定，相对于 Alpha 版已经有了很大的进步，消除了严重错误  
3、`rc`：和正式版基本相同，基本上不存在导致错误的 Bug  
4、`release`：最终版本  
:::

### 标识

`package.json`中版本号常有`^、~`或者`>=`等标识符:
:::tip
`插入符号^`：固定主版本号，次版本号和修订号可以随意更改，例如^2.0.0，可以使用 2.0.1、2.2.2 、2.9.9 的版本。  
`波浪符号~`：固定主版本号和次版本号，修订号可以随意更改，例如~2.0.0，可以使用 2.0.0、2.0.2 、2.0.9 的版本。  
`对比符号类的`：`>`(大于) `>=`(大于等于) `<`(小于) `<=`(小于等于)  
`或符号||`：可以用来设置多个版本号限制规则，例如 >= 3.0.0 || <= 1.0.0  
`没有任何符号`：完全百分百匹配，必须使用当前版本号  
`任意版本`：对版本没有限制，一般不用  
:::

## 禁用 lock 文件锁定版本

在 `.npmrc` 文件中，`package-lock=false` 是一条配置项，用于禁用 `npm` 的包锁文件 (`package-lock.json`) 的生成和使用。

`package-lock.json` 是 `npm 5` 之后引入的文件，用于记录当前安装的每个软件包的精确版本和依赖关系。它的存在旨在确保在不同环境中重复安装时，使用相同的依赖版本，从而提供可预测性和一致性。

将 `package-lock=false` 添加到 `.npmrc` 文件中，意味着你不希望生成或使用包锁文件。这可能会导致以下影响：

在执行 `npm install` 或 `npm ci` 时，不会生成新的 `package-lock.json` 文件。

如果项目中已经存在 `package-lock.json` 文件，npm 将不会使用它来安装软件包，而是依赖于 `package.json` 中的版本范围进行安装。在安装软件包时，npm 可能会使用更松散的版本范围，而不是精确匹配 `package-lock.json` 中的版本。

请注意，禁用包锁文件可能会导致一些问题，特别是在多人协作或部署到生产环境时。包锁文件的使用有助于确保团队成员使用相同的依赖版本，并提供可重复构建的保证。因此，在决定禁用包锁文件之前，请仔细**评估项目需求和潜在的风险**。
