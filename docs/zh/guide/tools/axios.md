---
title: Axios
---
[Axios官网](https://axios-http.com/zh/docs/intro)
### 创建一个请求实例
```js
const RequestService = axios.create({
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
### 拦截器
在请求或响应被 then 或 catch 处理前拦截它们。
```js
// 添加请求拦截器
RequestService.interceptors.request.use(function (config) {
  // 在发送请求之前做些什么
  return config;
}, function (error) {
  // 对请求错误做些什么
  return Promise.reject(error);
});

// 添加响应拦截器
RequestService.interceptors.response.use(function (response) {
  // 2xx 范围内的状态码都会触发该函数。
  // 对响应数据做点什么
  return response;
}, function (error) {
  // 超出 2xx 范围的状态码都会触发该函数。
  // 对响应错误做点什么
  return Promise.reject(error);
});
```
移除拦截器
```js
const myInterceptor = instance.interceptors.request.use(function () {/*...*/});
axios.interceptors.request.eject(myInterceptor);
```
### 取消请求(场景：快速点击Tab标签)
也可以通过传递一个 executor 函数到 CancelToken 的构造函数来创建一个 cancel token：
```js
const CancelToken = axios.CancelToken;
let cancel;
axios.get('/user/12345', {
  cancelToken: new CancelToken(function executor(c) {
    // executor 函数接收一个 cancel 函数作为参数
    cancel = c;
  })
});
// 取消请求
cancel();
```





