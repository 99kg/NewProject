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
SAVE_FACES = config.get("save_faces", False)  # 是否保存人脸图片

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

# 创建img目录
if SAVE_FACES:
    img_dir = os.path.join(os.getcwd(), 'img')
    os.makedirs(img_dir, exist_ok=True)  # 确保目录存在


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
    """生成统计摘要报告"""
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

        if is_final:
            f.write(f"[阈值设置]\n高峰阈值: {PEAK_THRESHOLD}人, 低峰阈值: {LOW_THRESHOLD}人\n")
            f.write(f"最高人数：{peak_count}人, 最低人数：{low_count}人\n")
            f.write("-----------------------------------------\n")

            f.write("高峰时间段：\n")
            if peak_periods:
                valid_peak_periods = [p for p in peak_periods if p[0] != p[1]]
                if valid_peak_periods:
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

            f.write("\n低峰时间段：\n")
            if low_periods:
                valid_low_periods = [p for p in low_periods if p[0] != p[1]]
                if valid_low_periods:
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
    txt_filename = f"{base_name}_summary.txt" if segment_percentage == 1.0 else f"{base_name}_summary_{int(segment_percentage * 100)}percent.txt"
    txt_folder = os.path.join('utils/data/logs', base_name, 'txt')
    txt_path = os.path.join(txt_folder, txt_filename)

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
        is_final=(segment_percentage == 1.0)
    )


def get_current_periods(in_peak, in_low, current_peak_start, current_low_start, current_time):
    """获取当前所有时间段（包括未完成的）"""
    current_peak_periods = []
    current_low_periods = []

    if in_peak and current_peak_start is not None:
        current_peak_periods.append((
            seconds_to_hms(current_peak_start),
            seconds_to_hms(current_time)
        ))

    if in_low and current_low_start is not None:
        current_low_periods.append((
            seconds_to_hms(current_low_start),
            seconds_to_hms(current_time)
        ))

    return current_peak_periods, current_low_periods


def process_video(input_video, output_path=None, args=None):
    """主处理函数：处理视频流并进行人数统计"""
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    # 初始化视频源
    if isinstance(input_video, str):
        logger.info(f"Processing {os.path.basename(input_video)}...")
        vs = cv2.VideoCapture(input_video)
        total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        vs = input_video
        total_frames = None

    # 创建输出文件夹
    video_base_name = "output" if not isinstance(input_video, str) else os.path.splitext(os.path.basename(input_video))[0]
    video_output_folder = os.path.join('utils/data/logs', video_base_name)
    csv_folder = os.path.join(video_output_folder, 'csv')
    txt_folder = os.path.join(video_output_folder, 'txt')
    os.makedirs(csv_folder, exist_ok=True)
    os.makedirs(txt_folder, exist_ok=True)

    if SAVE_FACES:
        face_dir = os.path.join(img_dir, video_base_name)
        os.makedirs(face_dir, exist_ok=True)

    fps_video = vs.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    W, H = None, None

    # 初始化跟踪器和计数器
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}
    totalFrames = 0
    total_count = 0
    events = []
    gender_count = {"Male": 0, "Female": 0, "Unknown": 0}
    age_count = {age: 0 for age in ageList}
    peak_count = 0
    low_count = None
    peak_periods = []
    low_periods = []
    prev_count = 0
    last_count_time = time.time()
    current_interval_start = time.time()
    temp_objects = []
    fps = FPS().start()
    processing_start_time = time.time()
    in_peak_period = False
    in_low_period = False
    current_peak_start = None
    current_low_start = None
    start_time_offset = 0.01
    segment_reports_done = []
    next_report_percentage = REPORT_PERCENTAGE
    face_counter = 1

    # 主循环
    while True:
        ret, frame = vs.read()
        if not ret:
            break

        # 调整帧尺寸并强制转换为RGB
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 关键修复：统一使用RGB

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        if output_path and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, 30, (W, H), True)

        status = "Waiting"
        rects = []
        current_video_time_sec = totalFrames / fps_video + start_time_offset

        # 检测帧处理
        if totalFrames % args.get("skip_frames", 30) == 0:
            status = "Detecting"
            trackers = []
            temp_objects = []

            # 人脸检测（使用RGB图像）
            faceBoxes = highlightFace(faceNet, frame)
            for faceBox in faceBoxes:
                x1, y1, x2, y2 = faceBox
                face_img = frame[max(0, y1 - 20):min(y2 + 20, H - 1), max(0, x1 - 20):min(x2 + 20, W - 1)]
                if face_img.size > 0 and SAVE_FACES:
                    filename = f"face_{face_counter}.jpg"
                    filepath = os.path.join(face_dir, filename)
                    cv2.imwrite(filepath, face_img)
                    face_counter += 1

            # 行人检测
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > args.get("confidence", 0.4):
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] != "person":
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    # 使用RGB图像初始化跟踪器
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)  # 关键修复：传入RGB而非BGR
                    trackers.append(tracker)

                    box_center = ((startX + endX) // 2, (startY + endY) // 2)
                    if faceBoxes:
                        for faceBox in faceBoxes:
                            face_area = (faceBox[2] - faceBox[0]) * (faceBox[3] - faceBox[1])
                            person_area = (endX - startX) * (endY - startY)
                            xA = max(startX, faceBox[0])
                            yA = max(startY, faceBox[1])
                            xB = min(endX, faceBox[2])
                            yB = min(endY, faceBox[3])
                            inter_area = max(0, xB - xA) * max(0, yB - yA)
                            overlap_ratio = inter_area / min(face_area, person_area) if min(face_area, person_area) > 0 else 0

                            if overlap_ratio > 0.1:
                                face = frame[max(0, faceBox[1] - 20):min(faceBox[3] + 20, frame.shape[0] - 1),
                                       max(0, faceBox[0] - 20):min(faceBox[2] + 20, frame.shape[1] - 1)]
                                if face.size > 0:
                                    gender, age = predict_gender_age(face, genderNet, ageNet, genderList, ageList, MODEL_MEAN_VALUES)
                                    temp_objects.append({
                                        "centroid": box_center,
                                        "gender": gender,
                                        "age": age,
                                        "timestamp": time.time()
                                    })
        else:
            # 跟踪帧处理
            status = "Tracking"
            for tracker in trackers:
                tracker.update(rgb)  # 使用RGB图像更新跟踪器
                pos = tracker.get_position()
                rects.append((int(pos.left()), int(pos.top()),
                              int(pos.right()), int(pos.bottom())))

        # 更新对象跟踪
        objects = ct.update(rects)
        current_count = len(objects)
        now = time.time()

        if current_count > peak_count:
            peak_count = current_count

        if current_count > 0:
            if low_count is None:
                low_count = current_count
            elif current_count < low_count:
                low_count = current_count

        # 高峰/低峰时间段检测
        if current_count >= PEAK_THRESHOLD:
            if not in_peak_period:
                in_peak_period = True
                current_peak_start = current_video_time_sec
        else:
            if in_peak_period:
                in_peak_period = False
                if current_peak_start < current_video_time_sec:
                    peak_periods.append((
                        seconds_to_hms(current_peak_start),
                        seconds_to_hms(current_video_time_sec)
                    ))
                current_peak_start = None

        if current_count <= LOW_THRESHOLD and current_count > 0:
            if not in_low_period:
                in_low_period = True
                current_low_start = current_video_time_sec
        else:
            if in_low_period:
                in_low_period = False
                if current_low_start < current_video_time_sec:
                    low_periods.append((
                        seconds_to_hms(current_low_start),
                        seconds_to_hms(current_video_time_sec)
                    ))
                current_low_start = None

        # 处理每个跟踪对象
        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)
            if to is None:
                to = TrackableObject(objectID, centroid)
                trackableObjects[objectID] = to
            else:
                to.centroids.append(centroid)

            if to.gender == "Unknown" or to.age == "Unknown":
                closest_temp = None
                min_dist = float('inf')
                for temp in temp_objects[:]:
                    if time.time() - temp["timestamp"] > 5:
                        temp_objects.remove(temp)
                        continue
                    dist = np.linalg.norm(np.array(centroid) - np.array(temp["centroid"]))
                    if dist < min_dist and dist < 100:
                        min_dist = dist
                        closest_temp = temp

                if closest_temp:
                    old_gender = to.gender
                    old_age = to.age
                    to.update_gender_age(closest_temp["gender"], closest_temp["age"])
                    temp_objects.remove(closest_temp)
                    if to.counted:
                        for event in events:
                            if event["objectID"] == to.objectID:
                                if event["gender"] == "Unknown" or event["age"] == "Unknown":
                                    event["gender"] = to.gender
                                    event["age"] = to.age
                                    if old_gender in gender_count:
                                        gender_count[old_gender] -= 1
                                    if old_age in age_count:
                                        age_count[old_age] -= 1
                                    if to.gender in gender_count:
                                        gender_count[to.gender] += 1
                                    if to.age in age_count:
                                        age_count[to.age] += 1
                                break

            if not to.counted:
                event_time = seconds_to_hms(current_video_time_sec)
                event = {
                    "objectID": to.objectID,
                    "time": event_time,
                    "gender": to.gender,
                    "age": to.age
                }
                events.append(event)
                to.counted = True
                total_count += 1
                if to.gender in gender_count:
                    gender_count[to.gender] += 1
                if to.age in age_count:
                    age_count[to.age] += 1

            trackableObjects[objectID] = to
            display_text = f"ID {to.objectID}"
            if to.gender != "Unknown" or to.age != "Unknown":
                display_text += f" - {to.gender}, {to.age}"
            cv2.putText(frame, display_text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        # 显示统计信息
        info = [
            (f"Total: {total_count}", (10, H - 20)),
            (f"Current: {current_count}", (10, H - 40))
        ]
        status_info = f"PEAK ({current_count})" if in_peak_period else f"LOW ({current_count})" if in_low_period else f"NORMAL ({current_count})"
        cv2.putText(frame, status_info, (W - 150, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for text, pos in info:
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if writer is not None:
            writer.write(frame)
        cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        totalFrames += 1
        fps.update()

        if config.get("Timer", False) and (time.time() - processing_start_time) > 28800:
            break

        # 分段报告处理
        if SEGMENT_REPORTING and total_frames is not None:
            current_percentage = totalFrames / total_frames
            if current_percentage >= next_report_percentage:
                segment_percentage = next_report_percentage
                logger.info(f"[REPORT] Generating {int(segment_percentage * 100)}% segment report")
                current_peak, current_low = get_current_periods(
                    in_peak_period, in_low_period,
                    current_peak_start, current_low_start,
                    current_video_time_sec
                )
                report_peak_periods = peak_periods.copy() + current_peak
                report_low_periods = low_periods.copy() + current_low
                output_segment_report(
                    video_name=input_video,
                    segment_percentage=segment_percentage,
                    events=events,
                    total_count=total_count,
                    gender_count=gender_count,
                    age_count=age_count,
                    peak_periods=report_peak_periods,
                    low_periods=report_low_periods,
                    peak_count=peak_count,
                    low_count=low_count
                )
                next_report_percentage += REPORT_PERCENTAGE
                segment_reports_done.append(segment_percentage)
                if next_report_percentage > 1.0:
                    next_report_percentage = 1.0

    # 清理资源
    fps.stop()
    elapsed = time.time() - processing_start_time
    mins, secs = divmod(elapsed, 60)
    logger.info(f"[INFO] Time elapsed: {int(mins)}m{secs:.2f}s")
    logger.info(f"[INFO] Avg FPS: {fps.fps():.2f}")

    # 输出CSV文件
    csv_filename = os.path.join('utils/data/logs', video_base_name, 'csv', f"{video_base_name}_counting_data.csv")
    log_data(events, video_name=input_video, csv_path=csv_filename)

    # 输出最终报告
    if SEGMENT_REPORTING:
        if 1.0 not in segment_reports_done:
            logger.info("[REPORT] Generating final 100% segment report")
            output_segment_report(
                video_name=input_video,
                segment_percentage=1.0,
                events=events,
                total_count=total_count,
                gender_count=gender_count,
                age_count=age_count,
                peak_periods=peak_periods,
                low_periods=low_periods,
                peak_count=peak_count,
                low_count=low_count
            )
    else:
        txt_filename = os.path.join('utils/data/logs', video_base_name, 'txt', f"{video_base_name}_summary.txt")
        generate_summary(
            total_entries=total_count,
            gender_count=gender_count,
            age_count=age_count,
            peak_periods=peak_periods,
            low_periods=low_periods,
            peak_count=peak_count,
            low_count=low_count,
            video_name=input_video,
            output_path=txt_filename,
            is_final=True
        )

    cv2.destroyAllWindows()
    vs.release()
    if writer is not None:
        writer.release()


def predict_gender_age(face_img, genderNet, ageNet, genderList, ageList, MODEL_MEAN_VALUES):
    """预测人脸图像的性别和年龄（修复版）"""
    if face_img.size == 0:
        return "Unknown", "Unknown"

    try:
        # 转换为RGB格式（兼容dlib）
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # 预处理图像
        blob = cv2.dnn.blobFromImage(
            face_rgb,
            1.0,
            (227, 227),
            MODEL_MEAN_VALUES,
            swapRB=False  # 已转换为RGB，无需再交换通道
        )

        # 性别预测
        genderNet.setInput(blob)
        gender_preds = genderNet.forward()
        gender = genderList[gender_preds[0].argmax()]

        # 年龄预测
        ageNet.setInput(blob)
        age_preds = ageNet.forward()
        age = ageList[age_preds[0].argmax()]

        return gender, age

    except Exception as e:
        logger.error(f"Gender/age prediction failed: {str(e)}")
        return "Unknown", "Unknown"


def highlightFace(net, frame, conf_threshold=0.2):
    """人脸检测（修复版）"""
    try:
        # 转换为RGB格式（兼容性修复）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 预处理
        blob = cv2.dnn.blobFromImage(
            frame_rgb,
            1.0,
            (300, 300),
            [104, 117, 123],
            swapRB=False,  # 已转换为RGB
            crop=False
        )

        net.setInput(blob)
        detections = net.forward()
        faceBoxes = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                # 计算边界框坐标
                x1 = int(detections[0, 0, i, 3] * frame.shape[1])
                y1 = int(detections[0, 0, i, 4] * frame.shape[0])
                x2 = int(detections[0, 0, i, 5] * frame.shape[1])
                y2 = int(detections[0, 0, i, 6] * frame.shape[0])

                # 确保坐标不越界
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)

                if x2 > x1 and y2 > y1:  # 验证有效性
                    faceBoxes.append([x1, y1, x2, y2])

        return faceBoxes

    except Exception as e:
        logger.error(f"Face detection failed: {str(e)}")
        return []


def people_counter():
    """主程序入口（修复版）"""
    args = parse_arguments()

    # 检查视频文件
    video_folder = config.get("video_folder", "")
    video_files = [
        os.path.join(video_folder, f)
        for f in os.listdir(video_folder)
        if f.lower().endswith((".mp4", ".avi", ".mov"))
    ]

    if not video_files:
        logger.error("No video files found in specified folder!")
        return

    # 记录总处理时间
    total_start = time.time()
    logger.info(f"[TIMING] Processing started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for video_file in video_files:
        try:
            # 初始化视频处理
            video_start = time.time()
            logger.info(f"[TIMING] Processing: {os.path.basename(video_file)}")

            # 生成输出路径
            output_filename = f"{os.path.splitext(os.path.basename(video_file))[0]}_output.mp4"
            output_path = os.path.join(config.get("output_folder", "./output"), output_filename)

            # 处理视频
            process_video(video_file, output_path, args)

            # 记录处理时间
            elapsed = time.time() - video_start
            logger.info(f"[TIMING] Completed in {elapsed:.2f} seconds")

        except Exception as e:
            logger.error(f"Failed to process {video_file}: {str(e)}")
            continue

    # 最终统计
    total_elapsed = time.time() - total_start
    logger.info(f"[SUMMARY] All videos processed in {total_elapsed:.2f} seconds")


if __name__ == "__main__":
    # 验证dlib是否禁用AVX
    try:
        import dlib

        if getattr(dlib, "DLIB_USE_AVX", True):
            logger.warning("WARNING: dlib was compiled with AVX support (may cause crashes on M3 Max)")
    except ImportError:
        logger.error("dlib not installed! Run: pip install dlib --no-binary=dlib")

    # 启动程序
    people_counter()