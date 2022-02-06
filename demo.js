const createHello = (nation, province) => `我来自${nation}-${province}`;
const geneNation = (nation) => (province) => createHello(nation, province);

const fromChina = geneNation("中国");
const fromUSA = geneNation("美国")

console.log(fromChina("湖南"))
console.log(fromChina("广东"))

console.log(fromUSA("洛杉矶"))

console.log(geneNation("银河系")("地球"))
console.log(geneNation("三体")("黑暗森林"))