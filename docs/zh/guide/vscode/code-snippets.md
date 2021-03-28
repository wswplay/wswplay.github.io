---
title: 代码片段(Code Snippets)
---
## Vscode中怎么自定义代码片段
1. 点击Vscode **左下方小齿轮** 设置图标；
2. 点击 **用户代码** 片段；
3. 下拉列表右侧点击 **新代码片段**；
4. 输入文件名称，回车enter，即可自动打开新建的文件，文件名一般为 ```xxx.code-snippets```;
5. 按要求输入你需要自定义的代码即可。

### 代码片段的存放位置
```
/Users/你的名字/Library/Application Support/code/User/snippets
```

### 假如用来生成Vue模板文件是这样的：
```json
"init vue template": {
  "scope": "javascript,typescript,vue",
  "prefix": "iv",
  "body": [
    "<template>",
    "\t<div class='${1:className}'>\n",
    "\t</div>",
    "</template>\n",
    "<script>\n",
    "export default {",
    "\tname: '${1:className}',",
    "\tdata() {",
    "\t\treturn {}",
    "\t},",
    "}",
    "</script>\n",
    "<style lang='less' scoped>\n",
    "</style>"
  ],
  "description": "init vue template"
}
```
```scope```，是命令的作用域，即什么类型的文件需要vscode提示本命令。    
```prefix```的值```iv``` 就是命令。当你光标落地，键入字符与命令相匹配时，就会自动提示。    
```body```，就是需要生成的内容。    
``\t``，就是一个缩进。    
```\n```，就是换行。    
```${1:xxx}```，占位符。这里会高亮，光标会选中```$符号```们变量的位置。

### 那么用 ```iv```命令生成的模板如下：
光标会选中第 2，10 行```className```的位置
```vue {2,10}
<template>
  <div class='className'>

  </div>
</template>

<script>

export default {
  name: 'className',
  data() {
    return {}
  },
}
</script>

<style lang='less' scoped>

</style>
```
## 代码片段命令可以传参吗
你猜。。。暂时没找到。:stuck_out_tongue:
