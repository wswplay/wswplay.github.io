---
title: Promise和async/await
---
## 定义
```Promise``` 是实现异步的一种方案。对象用于表示一个异步操作的最终完成 (或失败)及其结果值。

Promise返回的是一个未知的未来值，本质是异步。但通过这种包装，使得我们可以用同步的方式将未来的结果和现在的处理程序联系起来。它不会立即返回最终的值，而是会返回一个 promise，以便在未来某个时候把值交给使用者。

就像人类的承诺。

### 必然的3种状态
一个 Promise 必然处于以下几种状态之一：
1. 待定（pending）: 初始状态，既没有被兑现，也没有被拒绝。
2. 已兑现（fulfilled）: 意味着操作成功完成。
3. 已拒绝（rejected）: 意味着操作失败。

待定状态的 Promise 对象要么会通过一个值被兑现（fulfilled），要么会通过一个原因（错误）被拒绝（rejected）。当这些情况之一发生时，我们用 promise 的 then 方法排列起来的相关处理程序就会被调用。如果 promise 在一个相应的处理程序被绑定时就已经被兑现或被拒绝了，那么这个处理程序就会被调用，因此在完成异步操作和绑定处理方法之间不会存在竞争状态。

因为 Promise.prototype.then 和  Promise.prototype.catch 方法返回的是 promise， 所以它们可以被链式调用。

## Promise.resolve(value)
返回一个状态由给定value决定的Promise对象。如果该值是thenable(即，带有then方法的对象)，返回的Promise对象的最终状态由then方法执行决定；    
否则的话(该value为空，基本类型或者不带then方法的对象),返回的Promise对象状态为fulfilled，并且将该value传递给对应的then方法。    
通常而言，如果您不知道一个值是否是Promise对象，使用Promise.resolve(value) 来返回一个Promise对象,这样就能将该value以Promise对象形式使用。

#### resolve()本质作用
1. resolve()是用来表示promise的状态为fullfilled，相当于只是定义了一个有状态的Promise，但是并没有调用它；
2. promise调用then的前提是promise的状态为fullfilled；
3. 只有promise调用then的时候，then里面的函数才会被推入微任务中；

**需要注意的是**，立即resolve的 Promise 对象，是在本轮“事件循环”（event loop）的结束时执行执行，不是马上执行,也不是在下一轮“事件循环”的开始时执行。    
**原因**：传递到 then() 中的函数被置入了一个微任务队列，而不是立即执行，这意味着它是在 JavaScript 事件队列的所有运行时结束了，事件队列被清空之后，才开始执行。

**我的理解是**：    
同步代码执行完  ->  执行本次循环已入列的微任务(如Promise的then方法)  ->  执行本次循环已入列的宏任务(如setTimeout(fn,0))  ->  下个循环...

## reject
## Promise.all
## Promise.race
## 创建Promise
```js
const myFirstPromise = new Promise((resolve, reject) => {
  // ?做一些异步操作，最终会调用下面两者之一:
  //   resolve(someValue); // fulfilled
  //   reject("failure reason"); // rejected
});
```
想要某个函数拥有promise功能，只需让其返回一个promise即可。
```js
function myAsyncFunction(url) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("GET", url);
    xhr.onload = () => resolve(xhr.responseText);
    xhr.onerror = () => reject(xhr.statusText);
    xhr.send();
  });
};
```
## async/await

#### 参考
[MDN Promise](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Global_Objects/Promise)