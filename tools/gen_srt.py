import sys
import os
import whisper
import datetime


# 设置时间格式(SRT标准)
def format_timestamp(seconds: float) -> str:
  td = datetime.timedelta(seconds=int(seconds))
  ms = int((seconds % 1) * 1000)
  return f"{str(td).zfill(8)},{ms:03}"


# Whisper 转录 + 生成 SRT
def transcribe_to_srt(input_path: str,
                      output_path: str = None,
                      model_size: str = "large"):

  # 模型规格：medium, small, tiny, base, large
  model = whisper.load_model(model_size)
  print(f"Transcribing '{input_path}' using model '{model_size}'...")

  result = model.transcribe(input_path, task="translate", language="ja")
  segments = result["segments"]

  if not output_path:
    base, _ = os.path.splitext(input_path)
    output_path = base + ".srt"

  with open(output_path, "w", encoding="utf-8") as f:
    for i, seg in enumerate(segments, start=1):
      start = format_timestamp(seg["start"])
      end = format_timestamp(seg["end"])
      text = seg["text"].strip()
      f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

  print(f"✅ 字幕文件已保存: {output_path}")


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("❌ 用法: python gen_srt.py your_video.mp4")
    sys.exit(1)

  video_path = sys.argv[1]
  transcribe_to_srt(video_path)
