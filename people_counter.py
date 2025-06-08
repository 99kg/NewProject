from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from imutils.video import FPS
import numpy as np
import argparse
import datetime
import logging
import imutils
import time
import dlib
import json
import csv
import cv2
import os

logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
logger = logging.getLogger(__name__)

with open("utils/config.json", "r") as file:
    config = json.load(file)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female', 'Unknown']

prototxt_file = os.path.join(config.get("model_folder", "./detector"), "MobileNetSSD_deploy.prototxt")
model_file = os.path.join(config.get("model_folder", "./detector"), "MobileNetSSD_deploy.caffemodel")
face_prototxt_file = os.path.join(config.get("model_folder", "./detector"), "opencv_face_detector.pbtxt")
face_model_file = os.path.join(config.get("model_folder", "./detector"), "opencv_face_detector_uint8.pb")
age_prototxt_file = os.path.join(config.get("model_folder", "./detector"), "age_deploy.prototxt")
age_model_file = os.path.join(config.get("model_folder", "./detector"), "age_net.caffemodel")
gender_prototxt_file = os.path.join(config.get("model_folder", "./detector"), "gender_deploy.prototxt")
gender_model_file = os.path.join(config.get("model_folder", "./detector"), "gender_net.caffemodel")

net = cv2.dnn.readNetFromCaffe(prototxt_file, model_file)
faceNet = cv2.dnn.readNet(face_prototxt_file, face_model_file)
ageNet = cv2.dnn.readNet(age_prototxt_file, age_model_file)
genderNet = cv2.dnn.readNet(gender_prototxt_file, gender_model_file)


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", type=str, help="path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.4, help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=30, help="# of skip frames between detections")
    return vars(ap.parse_args())


def log_data(events, video_name="output"):
    if not events:
        logger.warning("No events to log.")
        return

    base_name = os.path.splitext(os.path.basename(video_name))[0]
    output_path = os.path.join('utils/data/logs', f"{base_name}_counting_data.csv")

    with open(output_path, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(("Event Type", "ID", "Time", "Gender", "Age"))

        for event in events:
            wr.writerow((
                event["event_type"],
                event["objectID"],
                event["time"],
                event["gender"],
                event["age"]
            ))


def generate_summary(total_entries, gender_count, age_count, peak_time, peak_count, low_time, low_count,
                     video_name="output"):
    base_name = os.path.splitext(os.path.basename(video_name))[0]
    output_path = os.path.join('utils/data/logs', f"{base_name}_summary.txt")

    # 定义儿童年龄组
    child_ages = ['(0-2)', '(4-6)', '(8-12)']
    children_count = sum(age_count.get(age, 0) for age in child_ages)

    with open(output_path, 'w') as f:
        f.write("---------------------------------------------------------------------\n")
        f.write(f"总人数：{total_entries}人\n")
        f.write(f"男性：{gender_count.get('Male', 0)}人\n")
        f.write(f"女性：{gender_count.get('Female', 0)}人\n")
        f.write(f"未知性别：{gender_count.get('Unknown', 0)}人\n")
        f.write(f"儿童：{children_count}人\n")
        f.write("---------------------------------------------------------------------\n")
        f.write(f"高峰时间段：{peak_time[0]}~{peak_time[1]}\n")
        f.write(f"高峰时间段最多人数：{peak_count}人\n")
        f.write(f"低峰时间段：{low_time[0]}~{low_time[1]}\n")
        f.write(f"低峰时间段最少人数：{low_count}人\n")
        f.write("---------------------------------------------------------------------\n")

    logger.info(f"Summary report generated: {output_path}")


def process_video(input_video, output_path=None, args=None):
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    # 初始化视频源
    if isinstance(input_video, str):
        logger.info(f"Processing {os.path.basename(input_video)}...")
        vs = cv2.VideoCapture(input_video)
    else:
        vs = input_video

    writer = None
    W, H = None, None

    # 初始化跟踪器
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    # 初始化计数器
    totalFrames = 0
    totalDown = 0
    totalUp = 0
    events = []  # 存储所有事件 - 使用字典列表

    # 初始化统计数据
    gender_count = {"Male": 0, "Female": 0, "Unknown": 0}
    age_count = {age: 0 for age in ageList}

    # 高峰和低谷统计
    peak_count = 0
    peak_start = None
    peak_end = None
    low_count = float('inf')
    low_start = None
    low_end = None

    # 时间段跟踪变量
    current_count = 0
    prev_count = 0
    last_count_time = start_time = time.time()
    current_interval_start = start_time

    # 临时对象存储
    temp_objects = []

    # 性能监控
    fps = FPS().start()

    while True:
        ret, frame = vs.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        if output_path and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, 30, (W, H), True)

        status = "Waiting"
        rects = []

        # 检测帧处理
        if totalFrames % args.get("skip_frames", 30) == 0:
            status = "Detecting"
            trackers = []
            temp_objects = []  # 每轮检测开始时清空临时对象

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

                    # 创建跟踪器
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)

                    # 计算检测框中心
                    box_center = ((startX + endX) // 2, (startY + endY) // 2)

                    # 人脸检测
                    _, faceBoxes = highlightFace(faceNet, frame)

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
                                    gender, age = predict_gender_age(face, genderNet, ageNet, genderList, ageList,
                                                                     MODEL_MEAN_VALUES)
                                    logger.info(f"[INFO] Detected face: {gender}, {age} in person box")

                                    # 存储为临时对象
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
                tracker.update(rgb)
                pos = tracker.get_position()
                rects.append((int(pos.left()), int(pos.top()),
                              int(pos.right()), int(pos.bottom())))

        # 绘制检测线
        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
        cv2.putText(frame, "-Prediction border - Entrance-", (10, H - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 更新对象跟踪
        objects = ct.update(rects)

        # 更新当前人数计数
        current_count = len(objects)
        now = time.time()

        # 更新高峰时间段
        if current_count > peak_count:
            peak_count = current_count
            peak_start = current_interval_start
            peak_end = now

        # 更新低峰时间段
        if current_count < low_count:
            low_count = current_count
            low_start = current_interval_start
            low_end = now

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
                        for event in reversed(events):
                            if event["objectID"] == objectID and event["event_type"] == "Move In":
                                if event["gender"] == "Unknown" or event["age"] == "Unknown":
                                    # 更新事件记录
                                    event["gender"] = to.gender
                                    event["age"] = to.age
                                    logger.info(f"[INFO] Updated event for ID:{objectID} to {to.gender}/{to.age}")

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

            # 进出事件检测
            if not to.counted:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)

                if direction < 0 and centroid[1] < H // 2:  # 离开
                    totalUp += 1
                    event_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                    event = {
                        "event_type": "Move Out",
                        "objectID": objectID,
                        "time": event_time,
                        "gender": to.gender,
                        "age": to.age
                    }
                    events.append(event)
                    to.counted = True
                    logger.info(f"[EVENT] Exit: ID {objectID} ({to.gender}, {to.age})")

                elif direction > 0 and centroid[1] > H // 2:  # 进入
                    totalDown += 1
                    event_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                    event = {
                        "event_type": "Move In",
                        "objectID": objectID,
                        "time": event_time,
                        "gender": to.gender,
                        "age": to.age
                    }
                    events.append(event)
                    to.counted = True
                    logger.info(f"[EVENT] Enter: ID {objectID} ({to.gender}, {to.age})")

                    # 统计性别和年龄（即使现在是Unknown，后续可能更新）
                    if to.gender in gender_count:
                        gender_count[to.gender] += 1
                    if to.age in age_count:
                        age_count[to.age] += 1

            trackableObjects[objectID] = to

            # 在画面上显示对象信息
            display_text = f"ID {objectID}"
            if to.gender != "Unknown" or to.age != "Unknown":
                display_text += f" - {to.gender}, {to.age}"
            cv2.putText(frame, display_text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        # 显示统计信息
        info = [
            (f"Exit: {totalUp}", (10, H - 20)),
            (f"Enter: {totalDown}", (10, H - 40)),
            (f"Status: {status}", (10, H - 60)),
            (f"Current: {current_count}", (10, H - 80))
        ]

        for text, pos in info:
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

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

        # 定时退出检查
        if config.get("Timer", False) and (time.time() - start_time) > 28800:
            break

    # 最终验证事件属性
    def validate_events(events):
        for event in events:
            if event["gender"] == "Unknown" or event["age"] == "Unknown":
                logger.warning(f"Event for ID {event['objectID']} still has unknown attributes: "
                               f"gender={event['gender']}, age={event['age']}")

    validate_events(events)

    # 清理资源
    fps.stop()
    logger.info(f"[INFO] Elapsed time: {time.time() - start_time:.2f}")
    logger.info(f"[INFO] Approx. FPS: {fps.fps():.2f}")

    # 记录所有事件到CSV
    log_data(events, video_name=input_video)

    # 生成总结报告
    generate_summary(
        total_entries=totalDown,
        gender_count=gender_count,
        age_count=age_count,
        peak_time=(
            datetime.datetime.fromtimestamp(peak_start).strftime("%H:%M:%S"),
            datetime.datetime.fromtimestamp(peak_end).strftime("%H:%M:%S")
        ) if peak_start and peak_end else ("00:00:00", "00:00:00"),
        peak_count=peak_count,
        low_time=(
            datetime.datetime.fromtimestamp(low_start).strftime("%H:%M:%S"),
            datetime.datetime.fromtimestamp(low_end).strftime("%H:%M:%S")
        ) if low_start and low_end else ("00:00:00", "00:00:00"),
        low_count=low_count,
        video_name=input_video
    )

    cv2.destroyAllWindows()
    vs.release()
    if writer is not None:
        writer.release()

def predict_gender_age(face_img, genderNet, ageNet, genderList, ageList, MODEL_MEAN_VALUES):
    if face_img.size == 0:
        return "Unknown", "Unknown"

    # 显示增强前后对比
    cv2.imshow("Original Face", face_img)

    try:
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        return gender, age
    except Exception as e:
        logger.error(f"Error in gender/age prediction: {str(e)}")
        return "Unknown", "Unknown"


def highlightFace(net, frame, conf_threshold=0.5, draw=False):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            if draw:
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)

    return frameOpencvDnn, faceBoxes


def people_counter():
    args = parse_arguments()
    video_folder = config.get("video_folder", "")
    video_files = [os.path.join(video_folder, f)
                   for f in os.listdir(video_folder)
                   if f.lower().endswith((".mp4", ".avi", ".mov")) and os.path.isfile(os.path.join(video_folder, f))]

    if video_files:
        for video_file in video_files:
            output_filename = os.path.splitext(os.path.basename(video_file))[0] + "_output.mp4"
            output_folder = config.get("output_folder", "./utils/data/logs")

            if output_folder.strip() == "":
                output_path = None
            else:
                output_path = os.path.join(output_folder, output_filename)

            process_video(video_file, output_path=output_path, args=args)
    else:
        logger.info("No video files found.")


if __name__ == "__main__":
    people_counter()
