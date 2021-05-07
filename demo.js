function sum() {
  let res = 0
  for (let i = 0; i < arguments.length; i++) {
    res += parseFloat(arguments[i]) || 0
  }
  return Number(res.toFixed(1))
}

console.log(sum(1, 2, 3, 4, 5))
console.log(sum(5, null, 5))
console.log(sum('1.0', false, 1, true, 1, 'A', 1, 'B', 1))
console.log(sum(0.1, 0.2))