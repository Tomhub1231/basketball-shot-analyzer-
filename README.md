# 🏀 篮球投篮动作分析工具
自动识别髋、膝、肩、肘4个关节角度，精准检测投篮出手点，输出慢放视频、角度CSV、角度变化图表。

## 📸 使用效果展示


## 环境依赖
Python3.11 + opencv-python + mediapipe + numpy + pillow + matplotlib + tqdm

## 使用方法
1. 下载 `shot.py` + `simhei.ttf` 放在同一文件夹
2. 运行 `python shot.py`
3. 自动生成 `input`、`output` 文件夹
4. 投篮视频放入 `input`，分析结果输出到 `output`
