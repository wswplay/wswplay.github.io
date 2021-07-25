---
title: Axios
---
### 创建一个请求实例
```js
const instance = axios.create({
  baseURL: 'https://some-domain.com/api/',
  timeout: 1000,
  headers: {'X-Custom-Header': 'foobar'}
});
```
### Requested请求配置
```js
{
  // 自定义请求头
  headers: {'X-Requested-With': 'XMLHttpRequest'},
  // `auth` HTTP Basic Auth
  auth: {
    username: 'janedoe',
    password: 's00pers3cret'
  },
  timeout: 1000, // 默认值是 `0` (永不超时)
  // `baseURL` 将自动加在 `url` 前面，除非 `url` 是一个绝对 URL。
  // 它可以通过设置一个 `baseURL` 便于为 axios 实例的方法传递相对 URL
  baseURL: 'https://some-domain.com/api/',
  url: '/demo', // 唯一的必填项
  method: 'post' // 默认为get
  params: { // URL参数，多用于get
    ID: 12345
  },
  data: { // 请求体数据，多用于post
    firstName: 'Fred'
  },
  // `responseType` 表示浏览器将要响应的数据类型
  // 选项包括: 'arraybuffer', 'document', 'json', 'text', 'stream'
  // 浏览器专属：'blob'
  responseType: 'json', // 默认值
  // 代理设置
  proxy: {
    protocol: 'https',
    host: '127.0.0.1',
    port: 9000,
    auth: {
      username: 'mikeymike',
      password: 'rapunz3l'
    }
  },
}
```