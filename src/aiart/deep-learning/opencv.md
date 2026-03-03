---
title: OpenCV
outline: deep
---

# OpenCV

OpenCV(Open Source Computer Vision) is the world's biggest computer vision library.

OpenCV is open source, contains over 2500 algorithms, and is operated by the non-profit Open Source Vision Foundation. Since June 2000.

## 概览

- **PyTorch ≈ 大脑**（思考、学习、决策的核心）  
  它负责“理解”图像内容：这是猫？这是车？目标物体在哪？姿态是什么？深度是多少？它能从数据中学习复杂的模式、泛化到没见过的情况、做端到端的预测。这是智能的“认知”部分。

- **OpenCV ≈ 四肢 + 感官执行器**（动手、感知、精细操作的工具箱）  
  它负责：
  - “眼睛”：读摄像头、读视频、读图片文件
  - “手”：resize、crop、颜色转换、归一化、letterbox、去畸变、画框、标注、alpha融合、图像拼接、实时显示
  - “基本反射”：传统快速算法（边缘检测、轮廓、透视变换、二值化、连通域、模板匹配、颜色阈值跟踪等），这些不需要“思考”，速度极快，适合实时前/后处理

```py
import cv2
import torch
import numpy as np

# OpenCV读图 + 预处理
img = cv2.imread("test.jpg")           # BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 转RGB
img_resized = cv2.resize(img, (640, 640))
img_tensor = torch.from_numpy(img_resized).permute(2,0,1).float() / 255.0
img_tensor = img_tensor.unsqueeze(0).cuda()   # [1,3,640,640]

# PyTorch模型推理
model = torch.load("yolov11n.pt")   # 或 ultralytics 的 YOLO类
pred = model(img_tensor)

# OpenCV把结果画回原图
for box in pred.boxes:
    x1,y1,x2,y2 = box.xyxy[0].cpu().int().tolist()
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

cv2.imshow("result", img)
cv2.waitKey(0)
```
