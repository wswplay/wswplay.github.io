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
console.log('tempList', tempList)