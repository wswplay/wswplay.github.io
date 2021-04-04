---
title: Break、Continue与Label
---
## Break
break，彻底的跳出整个循环。跳出整个for。例如铁人三项，就是罢赛了，不玩了，回家了。

## Continue
continue，跳出本次循环。越过一个item，去往下一个item。还是铁人三项，放弃当前项，直接到下个项目比赛去了。

## Label
标记，用于跳出双重循环。放弃铁人三项，也不回家，直接蹦迪去了。:rocket:
```js {4,13,21,30}
let num = 0, flag = 10;
for(i = 0; i < flag; i++) {
  for(j= 0; j < flag; j++) {
    if(i === 5 && j === 5) break
    num++
  }
}
console.log(num) // 95  (i === 5 && j === 5)阻止了j，但i将继续执行。
// 解析：如果不设置条件，总分是100。当i = 5时，j只执行了5次就跳出了。所以100-5=95

for(i = 0; i < flag; i++) {
  for(j= 0; j < flag; j++) {
    if(i === 5 && j === 5) continue
    num++
  }
}
console.log(num) // 99

outer: for(i = 0; i < flag; i++) {
  for(j= 0; j < flag; j++) {
    if(i === 5 && j === 5) break outer
    num++
  }
}
console.log(num) // 55 同时终止i 和 j。
// 解析：总分100。当i = 4时，j已经执行了50次；当i = 5时，j只执行5次，就跳出双重循环了。

outer: for(i = 0; i < flag; i++) {
  for(j= 0; j < flag; j++) {
    if(i === 5 && j === 5) continue outer
    num++
  }
}
console.log(num) // 95
```
