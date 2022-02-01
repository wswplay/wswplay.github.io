let youSet = new Set();
youSet.add(2022)
youSet.add('虎')
youSet.add('喵')
console.log(youSet, youSet.size)

console.log([...youSet])

for(let item of youSet) {
  console.log(item)
}
for(let [key, val] of youSet.entries()) {
  console.log(`${key}===${val}`)
}

for(let [key, val] of [...youSet].entries()) {
  console.log(`${key}===${val}`)
}

