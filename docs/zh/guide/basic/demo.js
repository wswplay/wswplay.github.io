const list = [
  {name: 'one', id: 1},
  {name: 'two'},
  {name: 'three', id: 3},
]

let tempList = []
list.forEach(item => {
  if(!item.id) return
  tempList.push(item)
})
// console.log('tempList', tempList)

const targetMap = new Map()
const person = {
  name: 'xiao',
  address: 'shenzhen'
}
targetMap.set(person, 'isYou')
person.address = 'beijng'
console.log('beijing', targetMap.has(person), targetMap)