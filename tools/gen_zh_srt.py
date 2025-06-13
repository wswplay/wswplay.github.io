import sys
import os
import argparse
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
  # print("更新Argos 翻译包索引...")
  # argostranslate.package.update_package_index()

  print("🔍 检查可用 Argos 翻译包...")
  available_packages = argostranslate.package.get_available_packages()
  if not available_packages:
    print("❌ 没有可用的 Argos 翻译包，请检查网络连接或手动安装。")
    sys.exit(1)
  else:
    print("✅ 找到可用 Argos 翻译包。")
    package_to_install = next(
      filter(lambda x: x.from_code == from_code and x.to_code == to_code, available_packages))
    print(f"🚀 启用/下载 Argos 翻译模型...")
    argostranslate.package.install_from_path(package_to_install.download())


# Whisper 转录 + Argos 翻译 + 生成 SRT
def transcribe_and_translate(
  input_path: str,
  audio_code='ja',
  output_path: str = None,
  model_size="large",
):
  model = whisper.load_model(model_size)

  print(f"🚀 Whisper 正在识别、翻译源语音为英文... ({input_path})")
  result = model.transcribe(input_path, task="translate", language=audio_code)

  if not output_path:
    base, _ = os.path.splitext(input_path)
    output_path = base + ".srt"

  print("🚀 Argos 正在将英文翻译为中文...")
  translate = argostranslate.translate
  with open(output_path, "w", encoding="utf-8") as f:
    for i, seg in enumerate(result["segments"], start=1):
      start = format_timestamp(seg["start"])
      end = format_timestamp(seg["end"])
      en_text = seg["text"].strip()
      zh_text = translate.translate(en_text, from_code, to_code).strip()
      f.write(f"{i}\n{start} --> {end}\n{en_text}\n{zh_text}\n\n")

  print(f"🎉 中文字幕已生成: {output_path}")


# 主入口
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate captions")
  parser.add_argument("input_path", help="Path to the input video/audio file")
  parser.add_argument("--audio_code", default="ja", help="Audio language code")
  parser.add_argument("--output_path", default=None, help="Output file path")
  parser.add_argument("--model_size", default="large", help="Model size")

  args = parser.parse_args()

  ensure_argos_model_installed(from_code, to_code)
  transcribe_and_translate(
    input_path=args.input_path,
    audio_code=args.audio_code,
    output_path=args.output_path,
    model_size=args.model_size,
  )
