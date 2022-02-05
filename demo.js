console.log('111111')

setTimeout(() => {
  console.log('2222222')
})

const myNextTick = Promise.resolve();
myNextTick.then(logger)

console.log('444444')

// 辅助内容
const msg = '333333333';
function logger() {
  console.log(msg)
}