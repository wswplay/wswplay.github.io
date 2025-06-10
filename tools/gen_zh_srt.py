import sys
import os
import whisper
import datetime
import argostranslate.package
import argostranslate.translate

# 全局变量
from_code = "en"
to_code = "zh"


# 设置时间格式(SRT标准)
def format_timestamp(seconds: float) -> str:
  td = datetime.timedelta(seconds=int(seconds))
  ms = int((seconds % 1) * 1000)
  return f"{str(td).zfill(8)},{ms:03}"


# 加载 Argos 翻译模型(自动安装)
def ensure_argos_model_installed(from_code="en", to_code="zh"):
  print("更新Argos 翻译包索引...")
  argostranslate.package.update_package_index()

  print("🔍 检查可用的 Argos 翻译包...")
  available_packages = argostranslate.package.get_available_packages()
  print(f"可用的翻译包数量: {len(available_packages)}")
  if not available_packages:
    print("❌ 没有可用的 Argos 翻译包，请检查网络连接或手动安装。")
    sys.exit(1)
  else:
    print("✅ 找到可用的 Argos 翻译包。")
    package_to_install = next(
        filter(lambda x: x.from_code == from_code and x.to_code == to_code,
               available_packages))
    print(f"🚀 开始下载 Argos 翻译模型...")
    argostranslate.package.install_from_path(package_to_install.download())


# Whisper 转录 + Argos 翻译 + 生成 SRT
def transcribe_and_translate(
    input_path: str,
    audio_code='ja',
    output_path: str = None,
    model_size="large",
):
  print(f"🚀 Whisper 正在识别源语音... ({input_path})")
  model = whisper.load_model(model_size)

  print("✅ 日语识别完成，开始翻译为中文字幕...")
  result = model.transcribe(input_path, task="translate", language=audio_code)
  print("✅ 日语识别完成，开始翻译为中文字幕...")

  if not output_path:
    base, _ = os.path.splitext(input_path)
    output_path = base + ".srt"

  translate = argostranslate.translate
  with open(output_path, "w", encoding="utf-8") as f:
    for i, seg in enumerate(result["segments"], start=1):
      start = format_timestamp(seg["start"])
      end = format_timestamp(seg["end"])
      jp_text = seg["text"].strip()
      zh_text = translate.translate(jp_text, from_code, to_code).strip()
      f.write(f"{i}\n{start} --> {end}\n{zh_text}\n\n")

  print(f"🎉 中文字幕已生成: {output_path}")


# 主入口
if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("❌ 用法: python gen_srt_zh_offline.py your_video.mp4 audio_code")
    sys.exit(1)

  ensure_argos_model_installed(from_code, to_code)
  print(sys.argv)
  transcribe_and_translate(sys.argv[1])
