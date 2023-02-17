import { useRoute } from "vitepress";
import { computed, onMounted, ref, watch } from "vue";

export function useRandomPeotry() {
  const peoList = [
    "人有悲欢离合，月有阴晴圆缺，此事古难全。但愿人长久，千里共婵娟。",
    "大江东去，浪淘尽，千古风流人物。",
    "十年生死两茫茫，不思量，自难忘。",
    "老夫聊发少年狂，左牵黄，右擎苍。",
    "谁道人生无再少？门前流水尚能西！休将白发唱黄鸡。",
    "雪沫乳花浮午盏，蓼茸蒿笋试春盘。人间有味是清欢。",
    "休对故人思故国，且将新火试新茶。诗酒趁年华。",
    "春宵一刻值千金，花有清香月有阴。歌管楼台声细细，秋千院落夜沉沉。",
    "试问岭南应不好，却道：此心安处是吾乡。",
    "竹杖芒鞋轻胜马，谁怕？一蓑烟雨任平生。回首向来萧瑟处，归去，也无风雨也无晴。",
    "尊前不用翠眉颦。人生如逆旅，我亦是行人。",
    "试问闲情都几许？一川烟草，满城风絮，梅子黄时雨。",
    "莫等闲，白了少年头，空悲切。",
    "人间四月芳菲尽，山寺桃花始盛开。",
    "当夏天过去后，还有鲜花未曾开放。",
    "待到山花烂漫时，她在丛中笑。",
  ];

  let pId = ref(0);
  const curPoe = computed(() => peoList[pId.value]);
  const routeInfo = useRoute();

  function changePoe() {
    pId.value = Math.floor(Math.random() * peoList.length);
  }

  onMounted(() => changePoe());
  watch(
    () => routeInfo.path,
    () => changePoe()
  );

  return {
    curPoe,
    changePoe,
  };
}
