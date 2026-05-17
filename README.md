# 🏀 篮球投篮动作分析工具
自动识别髋、膝、肩、肘4个关节角度，精准检测投篮出手点，输出慢放视频、角度CSV、角度变化图表。

## 📸 使用效果展示
![投篮分析效果](https://raw.githubusercontent.com/Tomhub1231/basketball-shot-analyzer/main/QQ%E6%B5%8F%E8%A7%88%E5%99%A8%E6%88%AA%E5%9B%BE20260517182157.png)

## 环境依赖
Python3.11 + opencv-python + mediapipe + numpy + pillow + matplotlib + tqdm

## 使用方法
1. 下载 `shot.py` + `simhei.ttf` 放在同一文件夹
2. 运行 `python shot.py`
3. 自动生成 `input`、`output` 文件夹
4. 投篮视频放入 `input`，分析结果输出到 `output`
