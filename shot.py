import cv2
import mediapipe as mp
import numpy as np
import csv
import glob
import os
import sys
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# ---------------------- 打包环境核心修复（解决MediaPipe模型找不到问题） ----------------------
if getattr(sys, 'frozen', False):
    # 打包后：将PyInstaller临时解压目录添加到Python路径
    # MediaPipe会自动在sys.path里找modules文件夹
    sys.path.insert(0, sys._MEIPASS)

# ---------------------- 适配打包环境的路径配置 ----------------------
# 获取程序运行的根目录（打包后exe所在目录）
def get_root_path():
    if getattr(sys, 'frozen', False):
        # 打包后的exe运行环境
        return os.path.dirname(sys.executable)
    else:
        # 开发环境
        return os.path.dirname(os.path.abspath(__file__))

ROOT_PATH = get_root_path()
INPUT_FOLDER = os.path.join(ROOT_PATH, "input")
OUTPUT_FOLDER = os.path.join(ROOT_PATH, "output")

# 确保input/output文件夹存在
for folder in [INPUT_FOLDER, OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# ---------------------- 核心配置 ----------------------
SHOT_SIDE_FORCE = 'R'  # 固定右手投篮
print(f"强制指定投篮侧：{'右手（右侧）' if SHOT_SIDE_FORCE == 'R' else '左手（左侧）'}")

# 角度文字配置（适配打包后的字体加载）
def get_font():
    try:
        # 优先加载系统字体（打包后兼容）
        if os.name == 'nt':
            # Windows系统优先找系统字体
            font_paths = [
                "C:/Windows/Fonts/simhei.ttf",
                "C:/Windows/Fonts/msyh.ttc",
                os.path.join(ROOT_PATH, "simhei.ttf")  # 备用：exe同目录的字体文件
            ]
            for font_path in font_paths:
                if os.path.exists(font_path):
                    return ImageFont.truetype(font_path, 28)
        else:
            # 非Windows系统
            font = ImageFont.truetype("/Library/Fonts/Arial.ttf" if os.uname().sysname == 'Darwin' else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
            return font
    except:
        # 兜底：默认字体
        return ImageFont.load_default(size=28)

font = get_font()

# ---------------------- MediaPipe姿态检测（完全兼容旧版本） ----------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 关节索引
LANDMARK_MAP = {
    'L_Shoulder': 11, 'L_Elbow': 13, 'L_Wrist': 15, 'L_Hip': 23, 'L_Knee': 25,
    'R_Shoulder': 12, 'R_Elbow': 14, 'R_Wrist': 16, 'R_Hip': 24, 'R_Knee': 26
}

SIDE_MAP = {
    'L': {'Shoulder': 'L_Shoulder', 'Elbow': 'L_Elbow', 'Wrist': 'L_Wrist', 'Hip': 'L_Hip', 'Knee': 'L_Knee'},
    'R': {'Shoulder': 'R_Shoulder', 'Elbow': 'R_Elbow', 'Wrist': 'R_Wrist', 'Hip': 'R_Hip', 'Knee': 'R_Knee'}
}

JOINT_CN = {'Shoulder': '肩部', 'Elbow': '肘部', 'Hip': '髋部', 'Knee': '膝部'}

# ---------------------- 工具函数 ----------------------
def calculate_joint_angle(p1, p2, p3):
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    p3 = np.array(p3, dtype=np.float64)
    vec1 = p1 - p2
    vec2 = p3 - p2
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8
    angle_rad = np.arccos(np.clip(dot / norm, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return np.clip(angle_deg, 0, 180)

def draw_joint_text(frame, text, pos, font, color=(0, 255, 0), bg=(0, 0, 0, 140)):
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
    overlay = Image.new('RGBA', pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    text_bbox = draw.textbbox(pos, text, font=font)
    draw.rectangle([text_bbox[0]-4, text_bbox[1]-4, text_bbox[2]+4, text_bbox[3]+4], fill=bg)
    draw.text(pos, text, font=font, fill=color + (255,))
    pil = Image.alpha_composite(pil, overlay)
    return cv2.cvtColor(np.array(pil.convert('RGB')), cv2.COLOR_RGB2BGR)

def get_landmark_coords(landmarks, landmark_name, frame_shape):
    h, w = frame_shape[:2]
    landmark = landmarks[LANDMARK_MAP[landmark_name]]
    return (int(landmark.x * w), int(landmark.y * h))

# ---------------------- 视频处理主函数 ----------------------
def process_basketball_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return

    video_basename = os.path.basename(video_path).split('.')[0]
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_time = total_frames / orig_fps
    print(f"视频总时长：{total_time:.2f}秒")

    slow_fps = max(orig_fps / 5, 1)
    output_video_path = os.path.join(OUTPUT_FOLDER, f"{video_basename}_shot_analyzed.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, slow_fps, (3 * orig_width, orig_height))

    output_csv_path = os.path.join(OUTPUT_FOLDER, f"{video_basename}_shot_angles.csv")
    output_chart_path = os.path.join(OUTPUT_FOLDER, f"{video_basename}_shot_chart.png")
    csv_header = ['Frame_Index', 'Time(s)', 'Shoulder(°)', 'Elbow(°)', 'Hip(°)', 'Knee(°)', 'Shot_Side']
    csv_data_buffer = []

    # matplotlib 中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图表
    fig, ax_plot = plt.subplots(1, 1, figsize=(12, 6), dpi=100)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.12)

    ax_plot.set_xlabel('时间 (秒)', fontsize=20)
    ax_plot.set_ylabel('关节角度 (°)', fontsize=20)
    ax_plot.set_xlim(0, total_time)
    ax_plot.set_ylim(0, 180)
    ax_plot.tick_params(axis='both', labelsize=16)

    # 曲线配置
    curve_config = {
        'Shoulder': {'color': 'red', 'x': [], 'y': []},
        'Elbow': {'color': 'green', 'x': [], 'y': []},
        'Hip': {'color': 'blue', 'x': [], 'y': []},
        'Knee': {'color': 'orange', 'x': [], 'y': []}
    }
    plot_lines = {}
    for joint_name, config in curve_config.items():
        line, = ax_plot.plot([], [], label=JOINT_CN[joint_name], color=config['color'], linewidth=3)
        plot_lines[joint_name] = line
    ax_plot.legend(fontsize=20, loc='upper right')
    canvas = FigureCanvas(fig)

    shot_side = SHOT_SIDE_FORCE
    print(f"投篮侧：{shot_side}")

    # ---------- 出手点检测相关变量 ----------
    time_history = []                     # 所有时间点
    elbow_angle_history = []              # 肘角度历史
    # 出手点检测缓冲区
    elbow_angles_buffer = []              # 最近15帧肘角度
    elbow_velocity_buffer = []            # 最近5帧角速度
    detected_release_times = []           # 出手点时间列表
    detected_release_frames = []          # 出手点帧索引
    chart_release_drawn = []              # 已添加竖线的时间
    release_text_artist = None            # 右上角文本框

    # 出手检测参数
    ELBOW_BUFFER_SIZE = 15
    VELOCITY_BUFFER_SIZE = 5
    ANGLE_THRESHOLD = 140                 # 肘角度阈值
    VELOCITY_SPIKE_THRESHOLD = 10         # 角速度峰值阈值
    VELOCITY_DROP_RATIO = 0.5             # 速度骤降比例

    pbar = tqdm(total=total_frames, desc=f"处理 {video_basename}")
    frame_idx = 0
    prev_elbow_angle = 0.0
    shot_detected = False                 # 避免重复标记同一出手点

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_copy = frame.copy()
        frame_shape = frame_copy.shape
        current_time = frame_idx / orig_fps
        current_angles = {'Shoulder': 0.0, 'Elbow': 0.0, 'Hip': 0.0, 'Knee': 0.0}

        frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        pose_results = pose_detector.process(frame_rgb)

        # 姿态检测和角度计算
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            side_joints = SIDE_MAP[shot_side]

            shoulder_coords = get_landmark_coords(landmarks, side_joints['Shoulder'], frame_shape)
            elbow_coords = get_landmark_coords(landmarks, side_joints['Elbow'], frame_shape)
            wrist_coords = get_landmark_coords(landmarks, side_joints['Wrist'], frame_shape)
            hip_coords = get_landmark_coords(landmarks, side_joints['Hip'], frame_shape)
            knee_coords = get_landmark_coords(landmarks, side_joints['Knee'], frame_shape)

            current_angles['Shoulder'] = round(calculate_joint_angle(hip_coords, shoulder_coords, elbow_coords), 1)
            current_angles['Elbow'] = round(calculate_joint_angle(shoulder_coords, elbow_coords, wrist_coords), 1)
            current_angles['Hip'] = round(calculate_joint_angle(shoulder_coords, hip_coords, knee_coords), 1)
            knee_down_coords = (knee_coords[0], knee_coords[1] + 50)
            current_angles['Knee'] = round(calculate_joint_angle(hip_coords, knee_coords, knee_down_coords), 1)

            mp_drawing.draw_landmarks(
                frame_copy, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=1)
            )

            # 角度标签
            frame_copy = draw_joint_text(frame_copy, f"{current_angles['Shoulder']}°", (shoulder_coords[0] + 15, shoulder_coords[1] - 35), font)
            frame_copy = draw_joint_text(frame_copy, f"{current_angles['Elbow']}°", (elbow_coords[0] + 15, elbow_coords[1]), font)
            frame_copy = draw_joint_text(frame_copy, f"{current_angles['Hip']}°", (hip_coords[0] - 100, hip_coords[1]), font)
            frame_copy = draw_joint_text(frame_copy, f"{current_angles['Knee']}°", (knee_coords[0] - 100, knee_coords[1] + 35), font)

            # 更新曲线数据
            for joint_name, angle in current_angles.items():
                curve_config[joint_name]['x'].append(current_time)
                curve_config[joint_name]['y'].append(angle)
                plot_lines[joint_name].set_data(curve_config[joint_name]['x'], curve_config[joint_name]['y'])
            ax_plot.relim()
            ax_plot.autoscale_view(scalex=False, scaley=True)

        # ---------- 出手点检测（基于肘角度变化率） ----------
        elbow_angle = current_angles['Elbow']
        elbow_angle_history.append(elbow_angle)
        time_history.append(current_time)

        # 更新肘角度缓冲区
        elbow_angles_buffer.append(elbow_angle)
        if len(elbow_angles_buffer) > ELBOW_BUFFER_SIZE:
            elbow_angles_buffer.pop(0)

        # 计算角速度
        if frame_idx > 0:
            velocity = elbow_angle - prev_elbow_angle
            elbow_velocity_buffer.append(velocity)
            if len(elbow_velocity_buffer) > VELOCITY_BUFFER_SIZE:
                elbow_velocity_buffer.pop(0)

        # 出手点检测逻辑（需要足够的数据）
        if (not shot_detected and len(elbow_velocity_buffer) >= VELOCITY_BUFFER_SIZE and
            len(elbow_angles_buffer) >= ELBOW_BUFFER_SIZE):
            max_velocity = max(elbow_velocity_buffer)
            # 条件：速度峰值超过阈值，最新速度骤降，且肘角度较大
            if (max_velocity > VELOCITY_SPIKE_THRESHOLD and
                elbow_velocity_buffer[-1] < elbow_velocity_buffer[-2] * VELOCITY_DROP_RATIO and
                elbow_angle > ANGLE_THRESHOLD):
                # 出手点定位在速度峰值时刻（取速度最大帧对应的时间）
                # 由于速度缓冲区是对应于帧间差，出手点大约在速度峰值帧的后一帧，这里简单取当前时间
                release_time = current_time
                # 避免1秒内重复标记
                if not any(abs(release_time - t) < 1.0 for t in detected_release_times):
                    detected_release_times.append(release_time)
                    detected_release_frames.append(frame_idx)
                    shot_detected = True  # 只标记第一个出手点（可以根据需要改为允许多次）
                    print(f"  🏀 检测到出手点：{release_time:.2f}秒 (肘角度 {elbow_angle:.1f}°)")
        else:
            # 如果已经检测到，可以继续但不重置，也可以重置以便检测后续动作
            # 这里选择一旦检测就锁定，适合单次投篮视频
            pass

        prev_elbow_angle = elbow_angle

        # ====== 视频帧出手点标注 ======
        for rt, rframe in zip(detected_release_times, detected_release_frames):
            if abs(frame_idx - rframe) <= 3:
                cv2.putText(frame_copy, f"RELEASE! {rt:.2f}s", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.circle(frame_copy, (orig_width // 2, orig_height // 2), 80, (0, 255, 255), 4)
                cv2.putText(frame_copy, "RELEASE", (orig_width // 2 - 80, orig_height // 2 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)

        # ====== 图表出手点标记 ======
        for rt in detected_release_times:
            if rt not in chart_release_drawn:
                ax_plot.axvline(x=rt, color='magenta', linestyle='--', linewidth=2, alpha=0.9)
                ax_plot.annotate(f'出手点\n{rt:.2f}s', xy=(rt, 170), xycoords='data',
                                 fontsize=11, color='magenta', fontweight='bold', ha='center',
                                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='magenta', alpha=0.9))
                chart_release_drawn.append(rt)

        # 右上角文本框
        if detected_release_times:
            times_str = ", ".join([f"{t:.2f}s" for t in sorted(detected_release_times)])
            if release_text_artist is not None:
                release_text_artist.remove()
            release_text_artist = ax_plot.text(0.98, 0.95, f"出手点: {times_str}",
                                               transform=ax_plot.transAxes,
                                               fontsize=12, color='magenta',
                                               verticalalignment='top', horizontalalignment='right',
                                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        else:
            if release_text_artist is not None:
                release_text_artist.remove()
                release_text_artist = None

        # ====== 动态 x 轴范围 ======
        if total_time <= 5.0:
            ax_plot.set_xlim(0, total_time)
        else:
            if current_time > 5:
                ax_plot.set_xlim(current_time - 5, current_time)
            else:
                ax_plot.set_xlim(0, 5)

        # 渲染图表并合成视频
        canvas.draw()
        plot_rgba = np.array(canvas.buffer_rgba())
        plot_bgr = cv2.cvtColor(plot_rgba, cv2.COLOR_RGBA2BGR)
        plot_resized = cv2.resize(plot_bgr, (2 * orig_width, orig_height))
        combined_frame = np.hstack([frame_copy, plot_resized])
        video_writer.write(combined_frame)

        csv_data_buffer.append([
            frame_idx, round(current_time, 3),
            current_angles['Shoulder'], current_angles['Elbow'],
            current_angles['Hip'], current_angles['Knee'],
            '右侧' if shot_side == 'R' else '左侧'
        ])

        frame_idx += 1
        pbar.update(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pbar.close()
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    # ====== 保存最终图表：恢复完整 x 轴并刷新 ======
    if time_history:
        ax_plot.set_xlim(0, max(time_history))
        canvas.draw()
    fig.savefig(output_chart_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # 保存 CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header)
        csv_writer.writerows(csv_data_buffer)

    print(f"\n✅ 视频处理完成：")
    print(f"  - 输出视频：{output_video_path}")
    print(f"  - 角度数据：{output_csv_path}")
    print(f"  - 角度图表：{output_chart_path}")
    if detected_release_times:
        print(f"  - 出手点：{[round(t,2) for t in detected_release_times]} 秒")
    else:
        print("  - 未检测到出手点（请确认投篮动作中肘关节是否快速伸展至>140°）")

# ---------------------- 批量处理入口 ----------------------
if __name__ == "__main__":
    # 支持的视频格式
    supported_formats = ['mp4', 'avi', 'mov', 'mkv', 'flv']
    input_video_paths = []
    for fmt in supported_formats:
        input_video_paths.extend(glob.glob(os.path.join(INPUT_FOLDER, f"*.{fmt}")))

    if not input_video_paths:
        print("⚠️ 请将视频放入程序同目录的 'input' 文件夹（支持mp4/avi/mov/mkv/flv）")
        print("⚠️ 程序将在10秒后退出...")
        import time
        time.sleep(10)
    else:
        print(f"📌 发现 {len(input_video_paths)} 个视频，开始处理...\n")
        for video_path in input_video_paths:
            process_basketball_video(video_path)
        print("✨ 所有视频处理完毕！结果在程序同目录的 'output' 文件夹。")
        print("✨ 程序将在10秒后退出...")
        import time
        time.sleep(10)