import os
import argparse
import whisper
import datetime


# 设置时间格式(SRT标准)
def format_timestamp(seconds: float) -> str:
  td = datetime.timedelta(seconds=int(seconds))
  ms = int((seconds % 1) * 1000)
  return f"{str(td).zfill(8)},{ms:03}"


# Whisper 转录 + 生成 SRT
def transcribe_to_srt(
  input_path: str,
  audio_code: str = "en",
  output_path: str = None,
  model_size: str = "large",
):
  # 加载模型
  model = whisper.load_model(model_size)
  print(f"Path: '{input_path}' (Model size: {model_size})...")

  # 初始化所有变量
  result_original = model.transcribe(input_path, language=audio_code, task="transcribe")
  result_translated = None

  # 只有当非英语时才翻译
  if audio_code != default_code:
    result_translated = model.transcribe(input_path, task="translate")

  # 智能选择原始文本来源
  segments_original = result_original["segments"]
  segments_translated = result_translated["segments"] if result_translated else result_original[
    "segments"]

  # 生成输出路径
  output_path = output_path or os.path.splitext(input_path)[0] + ".srt"

  # 写入SRT文件
  with open(output_path, "w", encoding="utf-8") as f:
    for i, (seg_orig, seg_trans) in enumerate(zip(segments_original, segments_translated), start=1):
      start = format_timestamp(seg_orig["start"])
      end = format_timestamp(seg_orig["end"])
      text_original = seg_orig["text"].strip()
      text_translated = seg_trans["text"].strip()

      f.write(f"{i}\n{start} --> {end}\n{text_original}" +
              (f"\n{text_translated}" if text_original != text_translated else "") + "\n\n")

  print(f"✅ 字幕文件已保存: {output_path}")


if __name__ == "__main__":
  default_code = "en"

  parser = argparse.ArgumentParser(description="Generate srt file")
  parser.add_argument("input_path", help="Path to the input video/audio file")
  parser.add_argument("--audio_code",
                      default=default_code,
                      help="ja,ko,zh,en,fr,es,pt,de,it,ru,ar,hi,th,id,vi,tr")
  parser.add_argument("--output_path", default=None, help="Output file path")
  parser.add_argument("--model_size", default="large", help="medium, small, tiny, base, large")

  args = parser.parse_args()
  transcribe_to_srt(
    input_path=args.input_path,
    audio_code=args.audio_code,
    output_path=args.output_path,
    model_size=args.model_size,
  )
