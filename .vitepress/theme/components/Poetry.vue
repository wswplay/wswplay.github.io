<template>
  <div class="poetry" @click="changePoe">{{ curPoe }}</div>
</template>

<script setup lang="ts">
import { useRandomPeotry } from "../composables/slotsData";
const { curPoe, changePoe } = useRandomPeotry();
</script>

<style scoped lang="less">
.poetry {
  padding: 15px;
  margin: 10px 0;
  width: 100%;
  border: 1px solid rgba(205, 203, 203, 0.5);
  border-radius: 5px;
  cursor: pointer;
  opacity: 0.6;
  transition: all 0.6s;
  &:hover {
    opacity: 0.8;
    transform: scale(1.02);
  }
}
</style>
