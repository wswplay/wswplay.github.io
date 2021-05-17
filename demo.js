const state = {
  world: {
    name: '世界',
    china: {
      name: '中国',
      vue: {
        name: 'Vue'
      }
    }
  }
}
const path = ['world', 'china']
let cname = path.reduce((res, item) => res[item], state)
// console.log('name=== ', cname)


function logger({name = 'xiao', address = 'shenzhen'} = {}) {
  return state => {
    if(state.world.china) {
      console.log(`我看看${name} ${address}`)
    } else {
      console.log('暂无数据')
    }
  }
}
logger()(state)

function miao(...arg) {
  console.log('arg', arg)
}
miao(6, 66, 699)