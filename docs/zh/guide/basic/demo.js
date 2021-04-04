let num = 0, flag = 10;
// for(i = 0; i < flag; i++) {
//   for(j= 0; j < flag; j++) {
//     // if(i === 5 && j === 5) break
//     if(i === 5 && j === 5) continue
//     num++
//   }
// }

outer: for(i = 0; i < flag; i++) {
  for(j= 0; j < flag; j++) {
    // if(i === 5 && j === 5) break outer
    console.log(i, '--', j)
    if(i === 5 && j === 5) continue outer
    num++
  }
}


console.log('num==: ', num)