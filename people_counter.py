# 导入必要的库和模块
from tracker.centroidtracker import CentroidTracker  # 质心跟踪器
from tracker.trackableobject import TrackableObject  # 可跟踪对象类
from imutils.video import FPS  # 帧率计数器
import numpy as np  # 数值计算库
import argparse  # 命令行参数解析
import datetime  # 日期时间处理
import logging  # 日志记录
import imutils  # 图像处理工具
import time  # 时间处理
import dlib  # 机器学习库（用于对象跟踪）
import json  # JSON数据处理
import csv  # CSV文件操作
import cv2  # OpenCV计算机视觉库
import os  # 操作系统接口

# 配置日志记录
logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger(__name__)

# 加载配置文件
with open("utils/config.json", "r") as file:
    config = json.load(file)

# 从配置中获取阈值和报告设置
PEAK_THRESHOLD = config.get("Peak_Threshold", 10)  # 高峰人数阈值，默认10
LOW_THRESHOLD = config.get("Low_Threshold", 5)  # 低峰人数阈值，默认5
SEGMENT_REPORTING = config.get("segment_reporting", False)  # 是否分段输出报告
REPORT_PERCENTAGE = config.get("report_percentage", 0.2)  # 报告输出百分比，默认20%

# 模型参数
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)  # 图像预处理均值
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']  # 年龄分组
genderList = ['Male', 'Female']  # 性别分类

# 构建模型文件路径
prototxt_file = os.path.join(config.get("model_folder", "./detector"), "MobileNetSSD_deploy.prototxt")
model_file = os.path.join(config.get("model_folder", "./detector"), "MobileNetSSD_deploy.caffemodel")
face_prototxt_file = os.path.join(config.get("model_folder", "./detector"), "opencv_face_detector.pbtxt")
face_model_file = os.path.join(config.get("model_folder", "./detector"), "opencv_face_detector_uint8.pb")
age_prototxt_file = os.path.join(config.get("model_folder", "./detector"), "age_deploy.prototxt")
age_model_file = os.path.join(config.get("model_folder", "./detector"), "age_net.caffemodel")
gender_prototxt_file = os.path.join(config.get("model_folder", "./detector"), "gender_deploy.prototxt")
gender_model_file = os.path.join(config.get("model_folder", "./detector"), "gender_net.caffemodel")

# 加载预训练模型
net = cv2.dnn.readNetFromCaffe(prototxt_file, model_file)  # 行人检测模型
faceNet = cv2.dnn.readNet(face_prototxt_file, face_model_file)  # 人脸检测模型
ageNet = cv2.dnn.readNet(age_prototxt_file, age_model_file)  # 年龄预测模型
genderNet = cv2.dnn.readNet(gender_prototxt_file, gender_model_file)  # 性别预测模型


def parse_arguments():
    """解析命令行参数"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", type=str, help="path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.4, help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=30, help="# of skip frames between detections")
    return vars(ap.parse_args())


def seconds_to_hms(seconds):
    """将秒数转换为HH:MM:SS格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def log_data(events, video_name="output", csv_path=None):
    """将事件记录到CSV文件"""
    if not events:
        logger.warning("No events to log.")
        return

    if csv_path is None:
        base_name = os.path.splitext(os.path.basename(video_name))[0]
        csv_path = os.path.join('utils/data/logs', base_name, 'csv', f"{base_name}_counting_data.csv")
    else:
        # 确保目录存在
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(("ID", "Time", "Gender", "Age"))

        for event in events:
            wr.writerow((
                event["objectID"],
                event["time"],
                event["gender"],
                event["age"]
            ))
    logger.info(f"Event log saved to: {csv_path}")


def generate_summary(total_entries, gender_count, age_count, peak_periods, low_periods,
                     peak_count, low_count, video_name="output", output_path=None, is_final=False):
    """生成统计摘要报告
    is_final: 是否为最终报告，如果是则输出阈值和高峰低峰时间段
    """
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(video_name))[0]
        output_path = os.path.join('utils/data/logs', base_name, 'txt', f"{base_name}_summary.txt")
    else:
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 定义儿童年龄组
    child_ages = ['(0-2)', '(4-6)', '(8-12)']
    children_count = sum(age_count.get(age, 0) for age in child_ages)

    with open(output_path, 'w') as f:
        f.write("------------------------------------------\n")
        f.write(f"总人数：{total_entries}人\n")
        f.write(f"男性：{gender_count.get('Male', 0)}人\n")
        f.write(f"女性：{gender_count.get('Female', 0)}人\n")
        f.write(f"未知性别：{gender_count.get('Unknown', 0)}人\n")
        f.write(f"儿童：{children_count}人\n")
        f.write("-----------------------------------------\n")

        # 如果是最终报告，则输出阈值设置和高峰低峰时间段
        if is_final:
            # 添加阈值信息
            f.write(f"[阈值设置]\n高峰阈值: {PEAK_THRESHOLD}人, 低峰阈值: {LOW_THRESHOLD}人\n")
            f.write(f"最高人数：{peak_count}人, 最低人数：{low_count}人\n")
            f.write("-----------------------------------------\n")

            # 输出所有高峰时间段
            f.write("高峰时间段：\n")
            if peak_periods:
                # 过滤掉无效时间段
                valid_peak_periods = [p for p in peak_periods if p[0] != p[1]]
                if valid_peak_periods:
                    # 去重处理
                    unique_peak_periods = []
                    for period in valid_peak_periods:
                        if period not in unique_peak_periods:
                            unique_peak_periods.append(period)

                    for period in unique_peak_periods:
                        start_time, end_time = period
                        f.write(f"{start_time}~{end_time}\n")
                else:
                    f.write("无有效高峰时间段\n")
            else:
                f.write("无高峰时间段\n")

            # 输出所有低峰时间段
            f.write("\n低峰时间段：\n")
            if low_periods:
                # 过滤掉无效时间段
                valid_low_periods = [p for p in low_periods if p[0] != p[1]]
                if valid_low_periods:
                    # 去重处理
                    unique_low_periods = []
                    for period in valid_low_periods:
                        if period not in unique_low_periods:
                            unique_low_periods.append(period)

                    for period in unique_low_periods:
                        start_time, end_time = period
                        f.write(f"{start_time}~{end_time}\n")
                else:
                    f.write("无有效低峰时间段\n")
            else:
                f.write("无低峰时间段\n")

        f.write("-----------------------------------------\n")

    logger.info(f"Summary report generated: {output_path}")


def output_segment_report(video_name, segment_percentage, events, total_count, gender_count,
                          age_count, peak_periods, low_periods, peak_count, low_count):
    """输出分段报告"""
    base_name = os.path.splitext(os.path.basename(video_name))[0]

    # 生成TXT报告文件名
    if segment_percentage == 1.0:
        txt_filename = f"{base_name}_summary.txt"
    else:
        txt_filename = f"{base_name}_summary_{int(segment_percentage * 100)}percent.txt"

    txt_folder = os.path.join('utils/data/logs', base_name, 'txt')
    txt_path = os.path.join(txt_folder, txt_filename)

    # 输出TXT报告
    generate_summary(
        total_entries=total_count,
        gender_count=gender_count,
        age_count=age_count,
        peak_periods=peak_periods,
        low_periods=low_periods,
        peak_count=peak_count,
        low_count=low_count,
        output_path=txt_path,
        video_name=video_name,
        is_final=(segment_percentage == 1.0)  # 仅当100%时标记为最终报告
    )


def get_current_periods(in_peak, in_low, current_peak_start, current_low_start, current_time):
    """获取当前所有时间段（包括未完成的）"""
    current_peak_periods = []
    current_low_periods = []

    # 添加当前高峰时间段（如果处于高峰）
    if in_peak and current_peak_start is not None:
        current_peak_periods.append((
            seconds_to_hms(current_peak_start),
            seconds_to_hms(current_time)
        ))

    # 添加当前低峰时间段（如果处于低峰）
    if in_low and current_low_start is not None:
        current_low_periods.append((
            seconds_to_hms(current_low_start),
            seconds_to_hms(current_time)
        ))

    return current_peak_periods, current_low_periods


def process_video(input_video, output_path=None, args=None):
    """主处理函数：处理视频流并进行人数统计"""
    # 定义目标检测类别
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    # 初始化视频源
    if isinstance(input_video, str):
        logger.info(f"Processing {os.path.basename(input_video)}...")
        vs = cv2.VideoCapture(input_video)
        # 获取视频总帧数
        total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        vs = input_video
        total_frames = None  # 实时视频流没有总帧数

    # 创建视频专用的输出文件夹
    video_base_name = "output"
    if isinstance(input_video, str):
        video_base_name = os.path.splitext(os.path.basename(input_video))[0]
    video_output_folder = os.path.join('utils/data/logs', video_base_name)
    csv_folder = os.path.join(video_output_folder, 'csv')
    txt_folder = os.path.join(video_output_folder, 'txt')
    os.makedirs(csv_folder, exist_ok=True)
    os.makedirs(txt_folder, exist_ok=True)

    # 获取视频帧率
    fps_video = vs.get(cv2.CAP_PROP_FPS)
    if fps_video <= 0:
        fps_video = 30.0  # 默认帧率
        logger.warning(f"Unable to get video FPS, using default: {fps_video}")

    writer = None
    W, H = None, None

    # 初始化跟踪器
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)  # 质心跟踪器
    trackers = []  # 跟踪器列表
    trackableObjects = {}  # 可跟踪对象字典

    # 初始化计数器
    totalFrames = 0  # 总帧数
    total_count = 0  # 总人数计数器
    events = []  # 存储所有事件 - 使用字典列表

    # 初始化统计数据
    gender_count = {"Male": 0, "Female": 0, "Unknown": 0}  # 性别统计
    age_count = {age: 0 for age in ageList}  # 年龄统计

    # 高峰和低谷统计
    peak_count = 0  # 峰值人数
    low_count = None  # 谷值人数（初始为None而不是0）

    # 存储所有高峰和低峰时间段
    peak_periods = []  # 每个元素为 (start_time, end_time) 格式为HH:MM:SS
    low_periods = []  # 每个元素为 (start_time, end_time) 格式为HH:MM:SS

    # 时间段跟踪变量
    prev_count = 0  # 上一帧人数
    last_count_time = time.time()  # 最后计数时间
    current_interval_start = time.time()  # 当前间隔开始时间

    # 临时对象存储（用于存储尚未匹配的检测结果）
    temp_objects = []

    # 性能监控
    fps = FPS().start()  # 帧率计数器

    # 记录处理开始时间
    processing_start_time = time.time()

    # 状态标志
    in_peak_period = False  # 是否处于高峰时段
    in_low_period = False  # 是否处于低峰时段
    current_peak_start = None  # 当前高峰开始时间
    current_low_start = None  # 当前低峰开始时间

    # 视频开始时间偏移量（跳过0秒）
    start_time_offset = 0.01  # 0.01秒，相当于00:00:01

    # 分段报告设置
    segment_reports_done = []  # 存储已完成的报告百分比
    next_report_percentage = REPORT_PERCENTAGE  # 下一个报告百分比

    # 主循环：处理视频每一帧
    while True:
        ret, frame = vs.read()
        if not ret:
            break

        # 调整帧尺寸并转换颜色空间
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 初始化帧尺寸
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # 初始化视频写入器
        if output_path and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, 30, (W, H), True)

        status = "Waiting"
        rects = []  # 检测框列表

        # 计算当前视频时间 (从视频开始经过的秒数) - 添加偏移量跳过0秒
        current_video_time_sec = totalFrames / fps_video + start_time_offset

        # 检测帧处理（每隔skip_frames帧进行一次目标检测）
        if totalFrames % args.get("skip_frames", 30) == 0:
            status = "Detecting"
            trackers = []  # 重置跟踪器
            temp_objects = []  # 每轮检测开始时清空临时对象

            # 关键修改：先进行人脸检测并绘制人脸框
            frame, faceBoxes = highlightFace(faceNet, frame, draw=True)

            # 使用MobileNet SSD进行目标检测
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            # 处理检测结果
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > args.get("confidence", 0.4):
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] != "person":
                        continue

                    # 获取边界框坐标
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    # 创建跟踪器
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)

                    # 计算检测框中心
                    box_center = ((startX + endX) // 2, (startY + endY) // 2)

                    # 人脸检测结果已经在上面的highlightFace调用中获取
                    if faceBoxes:
                        # 尝试匹配所有人脸
                        for faceBox in faceBoxes:
                            # 计算人脸框和人物框的重叠度
                            face_area = (faceBox[2] - faceBox[0]) * (faceBox[3] - faceBox[1])
                            person_area = (endX - startX) * (endY - startY)

                            # 计算交集
                            xA = max(startX, faceBox[0])
                            yA = max(startY, faceBox[1])
                            xB = min(endX, faceBox[2])
                            yB = min(endY, faceBox[3])

                            inter_area = max(0, xB - xA) * max(0, yB - yA)
                            overlap_ratio = inter_area / min(face_area, person_area) if min(face_area,
                                                                                            person_area) > 0 else 0

                            if overlap_ratio > 0.3:  # 重叠度超过30%
                                face = frame[max(0, faceBox[1] - 20):min(faceBox[3] + 20, frame.shape[0] - 1),
                                       max(0, faceBox[0] - 20):min(faceBox[2] + 20, frame.shape[1] - 1)]

                                if face.size > 0:
                                    # 预测性别和年龄
                                    gender, age = predict_gender_age(face, genderNet, ageNet, genderList, ageList,
                                                                     MODEL_MEAN_VALUES)
                                    logger.info(f"[INFO] Detected face: {gender}, {age} in person box")

                                    # 存储为临时对象
                                    temp_objects.append({
                                        "centroid": box_center,  # 中心点坐标
                                        "gender": gender,  # 预测性别
                                        "age": age,  # 预测年龄
                                        "timestamp": time.time()  # 时间戳
                                    })
        else:
            # 跟踪帧处理（在非检测帧使用跟踪器）
            status = "Tracking"
            for tracker in trackers:
                tracker.update(rgb)
                pos = tracker.get_position()
                rects.append((int(pos.left()), int(pos.top()),
                              int(pos.right()), int(pos.bottom())))

        # 更新对象跟踪
        objects = ct.update(rects)

        # 更新当前人数计数
        current_count = len(objects)
        now = time.time()

        # 更新整个视频的最大人数
        if current_count > peak_count:
            peak_count = current_count

        # 更新最低人数统计（忽略0值）
        if current_count > 0:  # 只考虑有人的情况
            if low_count is None:
                low_count = current_count
            elif current_count < low_count:
                low_count = current_count

        # 检查是否进入高峰时间段
        if current_count >= PEAK_THRESHOLD:
            if not in_peak_period:
                # 开始新的高峰时间段
                in_peak_period = True
                current_peak_start = current_video_time_sec
                logger.info(
                    f"[PEAK] Entering peak period ({current_count} >= {PEAK_THRESHOLD}) at {seconds_to_hms(current_peak_start)}")
        else:
            if in_peak_period:
                # 结束当前高峰时间段
                in_peak_period = False
                # 只记录有效时间段
                if current_peak_start < current_video_time_sec:
                    # 记录该高峰时间段（使用视频时间）
                    peak_periods.append((
                        seconds_to_hms(current_peak_start),
                        seconds_to_hms(current_video_time_sec)
                    ))
                    logger.info(
                        f"[PEAK] Leaving peak period ({current_count} < {PEAK_THRESHOLD}) at {seconds_to_hms(current_video_time_sec)}, Duration: {current_video_time_sec - current_peak_start:.2f}s")
                current_peak_start = None

        # 检查是否进入低峰时间段
        if current_count <= LOW_THRESHOLD and current_count > 0:  # 忽略0人情况
            if not in_low_period:
                # 开始新的低峰时间段
                in_low_period = True
                current_low_start = current_video_time_sec
                logger.info(
                    f"[LOW] Entering low period ({current_count} <= {LOW_THRESHOLD}) at {seconds_to_hms(current_low_start)}")
        else:
            if in_low_period:
                # 结束当前低峰时间段
                in_low_period = False
                # 只记录有效时间段
                if current_low_start < current_video_time_sec:
                    # 记录该低峰时间段（使用视频时间）
                    low_periods.append((
                        seconds_to_hms(current_low_start),
                        seconds_to_hms(current_video_time_sec)
                    ))
                    logger.info(
                        f"[LOW] Leaving low period ({current_count} > {LOW_THRESHOLD}) at {seconds_to_hms(current_video_time_sec)}, Duration: {current_video_time_sec - current_low_start:.2f}s")
                current_low_start = None

        # 重置间隔如果人数变化或超过5秒
        if current_count != prev_count or (now - last_count_time) > 5:
            current_interval_start = now
            prev_count = current_count

        last_count_time = now

        # 处理每个跟踪对象
        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)
                trackableObjects[objectID] = to

            else:
                # 更新现有对象位置
                to.centroids.append(centroid)

            # 尝试匹配临时对象（仅当属性未知或需要更新时）
            if to.gender == "Unknown" or to.age == "Unknown":
                closest_temp = None
                min_dist = float('inf')

                # 查找最近5秒内的临时对象
                for temp in temp_objects[:]:
                    # 检查时间有效性（5秒内）
                    if time.time() - temp["timestamp"] > 5:
                        temp_objects.remove(temp)
                        continue

                    dist = np.linalg.norm(np.array(centroid) - np.array(temp["centroid"]))
                    if dist < min_dist and dist < 100:  # 距离阈值100像素
                        min_dist = dist
                        closest_temp = temp

                if closest_temp:
                    old_gender = to.gender
                    old_age = to.age
                    to.update_gender_age(closest_temp["gender"], closest_temp["age"])
                    temp_objects.remove(closest_temp)
                    logger.info(f"[INFO] Matched temp object for ID:{objectID} - {to.gender}/{to.age}")

                    # 如果对象已经触发过事件，更新事件
                    if to.counted:
                        # 查找该对象最近的事件记录
                        for event in events:
                            if event["objectID"] == to.objectID:
                                if event["gender"] == "Unknown" or event["age"] == "Unknown":
                                    # 更新事件记录
                                    event["gender"] = to.gender
                                    event["age"] = to.age
                                    logger.info(f"[INFO] Updated event for ID:{to.objectID} to {to.gender}/{to.age}")

                                    # 更新统计信息：减去旧的Unknown计数，加上新的
                                    if old_gender in gender_count:
                                        gender_count[old_gender] -= 1
                                    if old_age in age_count:
                                        age_count[old_age] -= 1

                                    if to.gender in gender_count:
                                        gender_count[to.gender] += 1
                                    if to.age in age_count:
                                        age_count[to.age] += 1
                                break

            # 记录人员出现事件
            if not to.counted:
                event_time = seconds_to_hms(current_video_time_sec)
                event = {
                    "objectID": to.objectID,  # 使用对象的永久ID
                    "time": event_time,
                    "gender": to.gender,
                    "age": to.age
                }
                events.append(event)
                to.counted = True
                total_count += 1
                logger.info(f"[EVENT] ID {to.objectID} ({to.gender}, {to.age}) at {event_time}")

                # 统计性别和年龄
                if to.gender in gender_count:
                    gender_count[to.gender] += 1
                if to.age in age_count:
                    age_count[to.age] += 1

            trackableObjects[objectID] = to

            # 在画面上显示对象信息
            display_text = f"ID {to.objectID}"  # 显示对象的永久ID
            if to.gender != "Unknown" or to.age != "Unknown":
                display_text += f" - {to.gender}, {to.age}"
            cv2.putText(frame, display_text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        # 显示统计信息
        info = [
            (f"Total: {total_count}", (10, H - 20)),  # 总人数
            (f"Current: {current_count}", (10, H - 40))  # 当前人数
        ]

        # 添加当前状态信息
        status_info = ""
        if in_peak_period:
            status_info = f"PEAK ({current_count})"
        elif in_low_period:
            status_info = f"LOW ({current_count})"
        else:
            status_info = f"NORMAL ({current_count})"

        cv2.putText(frame, status_info, (W - 150, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for text, pos in info:
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 写入输出视频
        if writer is not None:
            writer.write(frame)

        # 显示画面
        cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        totalFrames += 1
        fps.update()

        # 定时退出检查（8小时超时）
        if config.get("Timer", False) and (time.time() - processing_start_time) > 28800:
            break

        # 分段报告处理
        if SEGMENT_REPORTING and total_frames is not None:
            current_percentage = totalFrames / total_frames
            # 检查是否需要生成分段报告
            if current_percentage >= next_report_percentage:
                # 生成当前进度的报告
                segment_percentage = next_report_percentage
                logger.info(f"[REPORT] Generating {int(segment_percentage * 100)}% segment report")

                # 获取当前时间段（包括未完成的）
                current_peak, current_low = get_current_periods(
                    in_peak_period, in_low_period,
                    current_peak_start, current_low_start,
                    current_video_time_sec
                )

                # 合并已完成和当前时间段
                report_peak_periods = peak_periods.copy() + current_peak
                report_low_periods = low_periods.copy() + current_low

                # 输出分段报告
                output_segment_report(
                    video_name=input_video,
                    segment_percentage=segment_percentage,
                    events=events,
                    total_count=total_count,
                    gender_count=gender_count,
                    age_count=age_count,
                    peak_periods=report_peak_periods,  # 使用合并后的时间段
                    low_periods=report_low_periods,
                    peak_count=peak_count,
                    low_count=low_count
                )

                # 更新下一个报告点
                next_report_percentage += REPORT_PERCENTAGE
                segment_reports_done.append(segment_percentage)

                # 确保不超过100%
                if next_report_percentage > 1.0:
                    next_report_percentage = 1.0

    # 处理结束前检查当前时间段
    if in_peak_period and current_peak_start is not None:
        # 确保结束时间大于开始时间
        if current_video_time_sec > current_peak_start:
            peak_periods.append((
                seconds_to_hms(current_peak_start),
                seconds_to_hms(current_video_time_sec)
            ))
            logger.info(
                f"[PEAK] Recording unfinished peak period at video end: {seconds_to_hms(current_peak_start)}~{seconds_to_hms(current_video_time_sec)}")

    if in_low_period and current_low_start is not None:
        # 确保结束时间大于开始时间
        if current_video_time_sec > current_low_start:
            low_periods.append((
                seconds_to_hms(current_low_start),
                seconds_to_hms(current_video_time_sec)
            ))
            logger.info(
                f"[LOW] Recording unfinished low period at video end: {seconds_to_hms(current_low_start)}~{seconds_to_hms(current_video_time_sec)}")

    # 最终验证事件属性
    def validate_events(events):
        """验证事件数据有效性"""
        for event in events:
            if event["gender"] == "Unknown" or event["age"] == "Unknown":
                logger.warning(f"Event for ID {event['objectID']} has unknown attributes: "
                               f"gender={event['gender']}, age={event['age']}")

    validate_events(events)

    # 确保事件数量与总人数一致
    if len(events) != total_count:
        logger.warning(f"事件数量({len(events)})与总人数({total_count})不一致，正在修复...")
        # 创建唯一对象ID列表
        unique_ids = set()
        for to in trackableObjects.values():
            if to.counted:
                unique_ids.add(to.objectID)

        # 更新总人数
        total_count = len(unique_ids)
        logger.info(f"修复后总人数: {total_count}人")

    # 清理资源
    fps.stop()  # 停止帧率计数器
    elapsed = time.time() - processing_start_time  # 计算总处理时间
    mins, secs = divmod(elapsed, 60)  # 将秒数转换为分钟和秒
    logger.info(f"[INFO] Time elapsed: {int(mins)}m{secs:.2f}s")  # 记录处理时间
    logger.info(f"[INFO] Avg FPS: {fps.fps():.2f}")  # 记录平均帧率

    # 输出CSV文件（无论是否分段报告都需要）
    csv_filename = os.path.join('utils/data/logs', video_base_name, 'csv', f"{video_base_name}_counting_data.csv")
    log_data(events, video_name=input_video, csv_path=csv_filename)  # 调用log_data函数保存事件数据

    # 输出最终报告（使用已经更新过的peak_periods和low_periods）
    if SEGMENT_REPORTING:
        # 检查是否已经输出了100%的报告
        if 1.0 not in segment_reports_done:
            logger.info("[REPORT] Generating final 100% segment report")
            output_segment_report(
                video_name=input_video,
                segment_percentage=1.0,
                events=events,
                total_count=total_count,
                gender_count=gender_count,
                age_count=age_count,
                peak_periods=peak_periods,  # 直接使用peak_periods
                low_periods=low_periods,  # 直接使用low_periods
                peak_count=peak_count,
                low_count=low_count
            )
    else:
        # 生成总结报告
        txt_filename = os.path.join('utils/data/logs', video_base_name, 'txt', f"{video_base_name}_summary.txt")
        generate_summary(  # 调用generate_summary函数生成统计报告
            total_entries=total_count,  # 总人数
            gender_count=gender_count,  # 性别统计
            age_count=age_count,  # 年龄统计
            peak_periods=peak_periods,  # 高峰时间段
            low_periods=low_periods,  # 低峰时间段
            peak_count=peak_count,  # 最高人数
            low_count=low_count,  # 最低人数
            video_name=input_video,  # 视频文件名
            output_path=txt_filename,
            is_final=True  # 标记为最终报告
        )

    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
    vs.release()  # 释放视频流资源
    if writer is not None:
        writer.release()  # 释放视频写入器资源


def predict_gender_age(face_img, genderNet, ageNet, genderList, ageList, MODEL_MEAN_VALUES):
    """预测给定人脸图像的性别和年龄"""
    if face_img.size == 0:  # 检查是否为空图像
        return "Unknown", "Unknown"  # 返回未知值

    # 显示已经识别的人脸
    # cv2.imshow("Original Face", face_img)

    try:
        # 预处理人脸图像
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # 预测性别
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]  # 获取最高概率的性别分类

        # 预测年龄
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]  # 获取最高概率的年龄分类

        return gender, age
    except Exception as e:
        logger.error(f"Gender/age prediction error: {str(e)}")  # 记录错误信息
        return "Unknown", "Unknown"  # 返回未知值


def highlightFace(net, frame, conf_threshold=0.5, draw=False):
    """在帧中检测人脸并可选地绘制边界框"""
    frameOpencvDnn = frame.copy()  # 创建帧的副本
    frameHeight = frameOpencvDnn.shape[0]  # 获取帧高度
    frameWidth = frameOpencvDnn.shape[1]  # 获取帧宽度

    # 预处理图像
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)  # 设置输入
    detections = net.forward()  # 进行人脸检测
    faceBoxes = []  # 存储人脸边界框

    # 处理检测结果
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # 获取置信度
        if confidence > conf_threshold:  # 过滤低置信度检测
            # 计算边界框坐标
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])  # 添加到人脸框列表

            if draw:  # 如果需要绘制边界框
                # 绘制白色矩形框（BGR格式：(255, 255, 255)是白色）
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (255, 255, 255), 2)  # 2是线宽
                # 添加文本标签
                label = f"Face: {confidence:.2f}"  # 显示置信度
                cv2.putText(frameOpencvDnn, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frameOpencvDnn, faceBoxes  # 返回带标记的帧和人脸框列表


def people_counter():
    """主函数：处理视频文件中的人流计数"""
    args = parse_arguments()  # 解析命令行参数

    # 获取视频文件夹路径
    video_folder = config.get("video_folder", "")
    # 列出所有支持的视频文件
    video_files = [os.path.join(video_folder, f)
                   for f in os.listdir(video_folder)
                   if f.lower().endswith((".mp4", ".avi", ".mov")) and os.path.isfile(os.path.join(video_folder, f))]

    if video_files:  # 如果有视频文件
        # 记录总体开始时间
        total_start_time = time.time()
        logger.info(f"[TIMING] 开始处理所有视频，时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        for video_file in video_files:  # 遍历每个视频文件
            # 记录单个视频开始时间
            video_start_time = time.time()
            logger.info(f"[TIMING] 开始处理视频: {os.path.basename(video_file)}")

            # 生成输出文件名
            output_filename = os.path.splitext(os.path.basename(video_file))[0] + "_output.mp4"
            output_folder = config.get("output_folder", "./utils/data/logs")  # 获取输出文件夹

            if output_folder.strip() == "":
                output_path = None  # 如果没有指定输出文件夹
            else:
                output_path = os.path.join(output_folder, output_filename)  # 完整的输出路径

            # 处理视频
            process_video(video_file, output_path=output_path, args=args)

            # 计算并输出单个视频处理时间
            video_elapsed = time.time() - video_start_time
            mins, secs = divmod(video_elapsed, 60)
            logger.info(f"[TIMING] 完成视频处理: {os.path.basename(video_file)} - 用时: {int(mins)}分{secs:.2f}秒")

        # 计算并输出所有视频总处理时间
        total_elapsed = time.time() - total_start_time
        total_mins, total_secs = divmod(total_elapsed, 60)
        logger.info(f"[TIMING] 所有视频处理完成 - 总用时: {int(total_mins)}分{total_secs:.2f}秒")

    else:
        logger.info("No video files found")  # 没有找到视频文件


if __name__ == "__main__":
    people_counter()  # 程序入口
