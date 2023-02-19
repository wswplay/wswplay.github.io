import { useRoute } from "vitepress";
import { computed, onMounted, ref, watch } from "vue";
import peotryData from "../peotryData";

export function useRandomPeotry() {
  let pId = ref(0);
  const curPoe = computed(() => peotryData[pId.value]);
  const routeInfo = useRoute();

  function changePoe() {
    pId.value = Math.floor(Math.random() * peotryData.length);
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
