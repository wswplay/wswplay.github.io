<script setup>
import { ref, onMounted } from "vue";

const isSpeaking = ref(false);
const statusText = ref("朗读全文");
const isSupported = ref(false);

// 获取Mac最佳中文语音（核心优化）
const getPremiumVoice = () => {
  const voices = window.speechSynthesis.getVoices();

  // 查找中文语音示例
  // const chineseVoices = voices.filter((v) => v.lang.includes("zh"));
  // console.log("中文语音:", chineseVoices);

  return (
    // voices.find((v) => v.voiceURI.includes("普通话")) ||
    voices.find((v) => v.voiceURI.includes("Tingting")) ||
    voices.find((v) => v.voiceURI.includes("Yu-shu")) || // Mac高清语音
    voices.find((v) => v.lang === "zh-CN") || // 其他中文
    voices[0] // 保底
  );
};

// 优化文本处理（解决机械音关键）
const getArticleText = () => {
  const mainContent = document.querySelector("main")?.innerText || "";

  return mainContent
    .replace(/([，。；！？])/g, "$1 ") // 中文标点后加空格
    .replace(/(\d+)/g, " $1 ") // 数字前后加空格
    .replace(/\s+/g, " ") // 合并多余空格
    .trim();
};

// 流畅朗读函数（您问的核心调用点）
const speakFluently = () => {
  const utterance = new SpeechSynthesisUtterance();
  utterance.text = getArticleText();
  utterance.voice = getPremiumVoice();

  // Mac专业调参（实测最佳）
  utterance.rate = 0.92;
  utterance.pitch = 1.08;
  utterance.volume = 0.95;

  // 事件监听
  // utterance.onboundary = (e) => {
  //   if (e.name === 'word') {
  //     console.log('正在朗读:', e.utterance.text.substring(e.charIndex, e.charIndex + e.charLength))
  //   }
  // }

  window.speechSynthesis.speak(utterance);
};

// 按钮控制主逻辑
const toggleSpeech = () => {
  if (!isSupported.value) {
    alert("您的浏览器不支持语音合成");
    return;
  }

  if (isSpeaking.value) {
    window.speechSynthesis.cancel();
    isSpeaking.value = false;
    statusText.value = "朗读全文";
  } else {
    isSpeaking.value = true;
    statusText.value = "停止朗读";
    speakFluently(); // 这里调用核心函数

    // 监听结束事件
    const checkEnd = setInterval(() => {
      if (!window.speechSynthesis.speaking) {
        isSpeaking.value = false;
        statusText.value = "朗读全文";
        clearInterval(checkEnd);
      }
    }, 500);
  }
};

// 初始化语音引擎（Mac专用）
onMounted(() => {
  if (!("speechSynthesis" in window)) return;

  let loadAttempt = 0;
  const loadVoices = () => {
    window.speechSynthesis.getVoices();
    if (loadAttempt++ < 5) {
      setTimeout(loadVoices, 300);
    } else {
      isSupported.value = window.speechSynthesis.getVoices().length > 0;
    }
  };

  // 双保险加载
  window.speechSynthesis.onvoiceschanged = loadVoices;
  loadVoices();
});
</script>

<template>
  <button
    @click="toggleSpeech"
    class="speech-btn"
    :class="{ speaking: isSpeaking }"
    :disabled="!isSupported"
  >
    {{ statusText }}
    <span class="icon">{{ isSpeaking ? "🔊" : "🔈" }}</span>
  </button>
</template>

<style scoped>
.speech-btn {
  padding: 4px;
  background: var(--vp-c-brand);
  color: #999;
  border: none;
  border-radius: 20px;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  transition: all 0.3s;
  margin: 1rem 0;
}

.speech-btn:hover {
  background: var(--vp-c-brand-dark);
  transform: translateY(-1px);
}

.speech-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.speech-btn.speaking {
  background: var(--vp-c-red);
  animation: pulse 1.5s infinite;
}

.icon {
  font-size: 16px;
}

@keyframes pulse {
  0% {
    opacity: 1;
  }
  50% {
    opacity: 0.7;
  }
  100% {
    opacity: 1;
  }
}
</style>
