// const demoData = [
//   {
//     server: { num: 0, type: ["128", "256"], mode: "nan-zhi", },
//     disk: [
//       { count: 0, size: "256GB" },
//       { count: 1, size: "256GB" },
//     ],
//   },
//   {
//     server: { num: 1, type: ["1024"], mode: "", },
//     disk: [
//       { dcount: 1, size: "256GB" },
//       { dcount: 2, size: "256GB" },
//     ],
//   },
// ];

// let flag = null;
// const needKeys = ["num", "count", "dcount", "type", "mode"];
// for(let item of demoData) {
//   let tempServer =  item.server;
//   for(let key in tempServer) {
//     if(needKeys.includes(key) && (!tempServer[key] || !String(tempServer[key]).trim())) {
//       flag = key;
//       console.log('server')
//       break;
//     }
//   }
//   let tempDist = item.disk;
//   for(let sit of tempDist) {
//     for(let sky in sit) {
//       if(needKeys.includes(sky) && (!tempServer[sky] || !String(tempServer[sky]).trim())) {
//         flag = sky;
//         console.log('disk')
//         break;
//       }
//     }
//   }
// }
// console.log('flag', flag);

// const demoList = [
//   { name: '', address: '0755'},
//   { name: 'dongguan', address: ''},
// ]
// waier:for(let item of demoList) {
//   for(let key in item) {
//     if(!item[key]) {
//       console.log('for-of', key);
//       break waier;
//     }
//   }
// }
// for(let i = 0; i < demoList.length; i++) {
//   let item = demoList[i];
//   for(let key in item) {
//     if(!item[key]) {
//       console.log('for', key)
//       break;
//     }
//   }
// }

// const demoObj = {
//   id: '666',
//   name: "shenzhen",
//   address: '0755',
//   mode: {
//     dep: '123',
//     prod: ""
//   },
//   lang: '',
//   os: 'nanzhi',
// }
// jumper:for(let key in demoObj) {
//   if(!demoObj[key]) {
//     console.log('objkey',key)
//     break;
//   }
//   console.log('key===', key);
//   let tempObj = demoObj[key];
//   if(tempObj instanceof Object) {
//     for(let sk in tempObj) {
//       if(!tempObj[sk]) {
//         console.log('sk', sk)
//         break;
//       }
//     }
//   }
// }
