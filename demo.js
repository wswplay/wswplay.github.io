const nanZhi = {
  id: "边城",
  address: "深圳",
};

const handler = {
  get(target, property) {
    return Reflect.get(...arguments);
  },
  set(target, property, value) {
    return Reflect.set(...arguments);
  }
};

const bianCheng = new Proxy(nanZhi, handler);

console.log(bianCheng.id);
console.log(bianCheng.nid);
bianCheng.id = "沈从文";
bianCheng.nid = "看过许多地方的云";
console.log(bianCheng.id);
console.log(bianCheng.nid);
