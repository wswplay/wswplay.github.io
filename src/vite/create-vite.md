---
title: Vite是怎么搭建生成一个项目的
# outline: deep
---

# Vite 是怎么搭建生成一个项目的

Vite 官方文档中，[搭建第一个 Vite 项目](https://cn.vitejs.dev/guide/#scaffolding-your-first-vite-project)，运行如下命令能创建一个项目，什么原理？

```bash
npm create vite@latest my-vue-app -- --template vue
```

## `npm create x` 等于 `npx create-x`

【[参考 npm 文档](http://nodejs.cn/npm/cli/v8/commands/npm-init/#forwarding-additional-options)】  
【[npx 即 npm exec](/node/npx.html)】

```bash
npm init <package-spec> (same as `npx <package-spec>)
npm init <@scope> (same as `npx <@scope>/create`)
# aliases: create, innit

# 举例子。注意后面是 npx(npm exec)
npm init foo -> npx create-foo
npm init @usr -> npx @usr/create
npm init @usr/foo -> npx @usr/create-foo
# 附加选项都将直接传递给命令
npm init foo -- --hello -> npx -- create-foo --hello
```

因此，`npm create vite@latest my-vue-app -- --template vue` 实际上等于：

```bash
npx create-vite@latest my-vue-app -- --template vue
```

**说人话就是**：临时安装 [create-vite](https://github.com/vitejs/vite/tree/main/packages/create-vite) 包，并执行 `create-vite` 命令，用后即删。

## create-vite 命令源码分析

直接调试、分析 ts 代码。【[参考：VSCode 调试 ts 文件](/core/typescript/tsx.html)】

> 进入 vite/packages/create-vite ，调试如下命令：
> `tsx src/index.ts my-vue-app --template vue`

```ts
// 简要版
init().catch((e) => {
  console.error(e);
});
async function init() {
  const argTargetDir = formatTargetDir(argv._[0]); // my-vue-app
  const argTemplate = argv.template || argv.t; // vue
  let targetDir = argTargetDir || defaultTargetDir; // my-vue-app
  const root = path.join(cwd, targetDir); // /vite/packages/create-vite/my-vue-app

  if (overwrite) {
    emptyDir(root);
  } else if (!fs.existsSync(root)) {
    // 这里创建my-vue-app文件夹
    fs.mkdirSync(root, { recursive: true });
  }

  let template: string = variant || framework?.name || argTemplate; // vue
  const pkgManager = pkgInfo ? pkgInfo.name : "npm"; // 没有传参，默认为npm
  const templateDir = path.resolve(
    fileURLToPath(import.meta.url),
    "../..",
    `template-${template}`
  ); // /vite/packages/create-vite/template-vue

  // 读template-vue模板文件，写入到my-vue-app
  const write = (file: string, content?: string) => {
    const targetPath = path.join(root, renameFiles[file] ?? file);
    if (content) {
      fs.writeFileSync(targetPath, content);
    } else {
      copy(path.join(templateDir, file), targetPath);
    }
  };
  const files = fs.readdirSync(templateDir);
  for (const file of files.filter((f) => f !== "package.json")) {
    write(file);
  }

  // 读取、修改默认package.json，写入到my-vue-app
  const pkg = JSON.parse(
    fs.readFileSync(path.join(templateDir, `package.json`), "utf-8")
  );
  pkg.name = packageName || getProjectName();
  write("package.json", JSON.stringify(pkg, null, 2) + "\n");

  // 最后确认路径，打印提示信息
  const cdProjectName = path.relative(cwd, root);
  console.log(`\nDone. Now run:\n`);
  if (root !== cwd) {
    console.log(
      `  cd ${
        cdProjectName.includes(" ") ? `"${cdProjectName}"` : cdProjectName
      }`
    );
  }
  switch (pkgManager) {
    case "yarn":
      console.log("  yarn");
      console.log("  yarn dev");
      break;
    default:
      console.log(`  ${pkgManager} install`);
      console.log(`  ${pkgManager} run dev`);
      break;
  }
  console.log();
}
```

**就一个函数，完结。**  
实际上就是，根据入参模板 `--template` 信息，把 `create-vite` 项目中对应模板目录内文件，全部复制到用户创建的新文件夹中(`package.json` 项目名称修改为入参)。
