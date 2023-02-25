---
title: Nodejs process模块及方法介绍与使用
---

# Process：进程

process 模块用来与当前进程互动，可以通过全局变量 process 访问，不必使用 require 命令加载。它是一个 EventEmitter 对象的实例。

## process.cwd()

`Current Work Directory` 的缩写。返回运行当前脚本的工作目录的路径。

## process.env

process.env 属性返回包含用户环境的对象。

```js
{
  TERM_SESSION_ID: 'w0t1p0:D1CF3869-8BBB-42D8-BED1-796B8CCEAC43',
  SSH_AUTH_SOCK: '/private/tmp/com.apple.launchd.pFTDx4IXYj/Listeners',
  LC_TERMINAL_VERSION: '3.4.19',
  COLORFGBG: '7;0',
  ITERM_PROFILE: 'Default',
  XPC_FLAGS: '0x0',
  LANG: 'zh_CN.UTF-8',
  PWD: '/Users/youraccount/mygithub/wswplay.github.io',
  SHELL: '/bin/zsh',
  __CFBundleIdentifier: 'com.googlecode.iterm2',
  TERM_PROGRAM_VERSION: '3.4.19',
  TERM_PROGRAM: 'iTerm.app',
  PATH: '/Users/youraccount/Library/pnpm:/Users/youraccount/.nvm/versions/node/v18.14.0/bin:/usr/local/bin:/System/Cryptexes/App/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin',
  LC_TERMINAL: 'iTerm2',
  COLORTERM: 'truecolor',
  COMMAND_MODE: 'unix2003',
  TERM: 'xterm-256color',
  HOME: '/Users/youraccount',
  TMPDIR: '/var/folders/9p/shsm7tgd14j_vy9zz2gb7y2r0000gn/T/',
  USER: 'youraccount',
  XPC_SERVICE_NAME: '0',
  LOGNAME: 'youraccount',
  ITERM_SESSION_ID: 'w0t1p0:D1CF3869-8BBB-42D8-BED1-796B8CCEAC43',
  __CF_USER_TEXT_ENCODING: '0x1F5:0x19:0x34',
  SHLVL: '1',
  OLDPWD: '/Users/youraccount',
  ZSH: '/Users/youraccount/.oh-my-zsh',
  PAGER: 'less',
  LESS: '-R',
  LSCOLORS: 'Gxfxcxdxbxegedabagacad',
  NVM_DIR: '/Users/youraccount/.nvm',
  NVM_CD_FLAGS: '-q',
  NVM_BIN: '/Users/youraccount/.nvm/versions/node/v18.14.0/bin',
  NVM_INC: '/Users/youraccount/.nvm/versions/node/v18.14.0/include/node',
  PNPM_HOME: '/Users/youraccount/Library/pnpm',
  _: '/Users/youraccount/.nvm/versions/node/v18.14.0/bin/node'
}
```
