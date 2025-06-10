import cv2
import numpy as np
import os
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input


class UTKFacePredictor:
    def __init__(self, model_path):
        """
        初始化UTKFace预测器
        :param model_path: 模型文件路径
        """
        # 加载模型并打印结构信息
        self.model = load_model(model_path)
        print(f"模型加载成功: {os.path.basename(model_path)}")

        # 打印模型结构信息
        print("\n" + "=" * 50)
        print("模型结构摘要:")
        self.model.summary()
        print("=" * 50 + "\n")

        # 获取模型输入尺寸
        self.input_shape = self.model.input_shape[1:3]  # (高度, 宽度)
        print(f"模型输入尺寸: {self.input_shape} ")

    def diagnose_output_layers(self):
        """诊断模型输出层结构"""
        print("输出层诊断:")
        for i, layer in enumerate(self.model.outputs):
            print(f"输出层 {i}: {layer.name}, 形状: {layer.shape}")

        # 检查输出层数量
        if len(self.model.outputs) < 2:
            print("警告: 模型输出层少于2个，可能无法正确预测年龄和性别")
        else:
            # 检查性别输出维度
            gender_output_shape = self.model.outputs[0].shape
            if gender_output_shape[-1] == 2:
                print("性别输出层: 检测到2个输出节点 (男/女)")
            else:
                print(f"警告: 性别输出层有异常维度 {gender_output_shape} ")

            # 检查年龄输出维度
            age_output_shape = self.model.outputs[1].shape
            if age_output_shape[-1] > 1:
                print(f"年龄输出层: 检测到{age_output_shape[-1]}个输出节点 (年龄分布)")
            else:
                print(f"警告: 年龄输出层有异常维度 {age_output_shape} ")

    def predict(self, face_img):
        """
        预测人脸图像的年龄和性别
        :param face_img: 人脸图像 (OpenCV格式)
        :return: (性别, 年龄)
        """
        if face_img.size == 0:
            return "Unknown", "Unknown"

        try:
            # 1. 将BGR转换为RGB（模型在RGB上训练）
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            print(f"RGB图像尺寸: {face_img_rgb.shape}, 数据类型: {face_img_rgb.dtype}")

            # 2. 根据模型要求调整尺寸
            img = cv2.resize(face_img_rgb, self.input_shape)
            print(f"调整后的图像尺寸: {img.shape}")

            # 3. 应用EfficientNet专用预处理
            img = preprocess_input(img.astype(np.float32))
            print(f"预处理后的图像数据范围: {img.min()} - {img.max()}")

            # 4. 扩展维度 (添加batch维度)
            img = np.expand_dims(img, axis=0)
            print(f"扩展后的图像尺寸: {img.shape}")

            # 5. 预测
            start_time = time.time()
            predictions = self.model.predict(img)
            print(f"预测耗时: {time.time() - start_time:.4f}秒")
            print(f"预测输出: {predictions}")

            # 解析预测结果
            return self.process_predictions(predictions)

        except Exception as e:
            print(f"预测过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return "Unknown", "Unknown"

    def process_predictions(self, predictions):
        gender = "Unknown"
        age = "Unknown"

        # 打印所有输出形状
        print(f"模型输出数量: {len(predictions)}")
        for i, pred in enumerate(predictions):
            print(f"输出层 {i} 形状: {pred.shape}, 内容: {pred}")

        # 性别预测
        if len(predictions) >= 1:
            gender_output = predictions[0]
            print(f"性别输出: {gender_output}")
            if gender_output.shape[-1] == 2:
                gender_probs = gender_output[0]
                gender = "Male" if gender_probs[0] > gender_probs[1] else "Female"
                print(f"性别预测: 男性概率={gender_probs[0]:.4f}, 女性概率={gender_probs[1]:.4f}")
            elif gender_output.shape[-1] == 1:
                gender = "Male" if gender_output[0][0] > 0.5 else "Female"
                print(f"性别预测(单输出): {gender_output[0][0]:.4f}")

        # 年龄预测
        if len(predictions) >= 2:
            age_output = predictions[1]
            print(f"年龄输出: {age_output}")
            if age_output.shape[-1] > 1:
                age_bins = np.arange(0, age_output.shape[-1])
                expected_age = np.sum(age_output[0] * age_bins)
                age = f"{expected_age:.1f}岁"
                print(f"期望年龄: {expected_age:.1f}岁")
            elif age_output.shape[-1] == 1:
                age = f"{age_output[0][0]:.1f}岁"
                print(f"回归年龄: {age_output[0][0]:.1f}岁")

        return gender, age


def process_images_in_folder(folder_path, predictor):
    """
    处理指定文件夹中的所有图片
    :param folder_path: 图片文件夹路径
    :param predictor: 预测器实例
    """
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    # 遍历文件夹中的所有文件
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)

        # 检查是否为图片文件
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
            try:
                print(f"\n{'=' * 50}")
                print(f"处理图片: {filename}")

                # 读取图片
                img = cv2.imread(file_path)
                if img is None:
                    print(f"无法读取图片: {file_path}")
                    continue

                # 复制图片用于显示（保留原始图片）
                display_img = img.copy()

                # 预测性别和年龄
                gender, age = predictor.predict(img)

                # 在图片上添加预测结果（白色文字）
                text = f"{gender}, {age}"
                cv2.putText(display_img, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # 显示图片
                cv2.imshow("Prediction Result", display_img)
                print(f"预测结果: {text}")

                # 等待按键
                key = cv2.waitKey(0)
                if key == 27:  # ESC键退出
                    break

            except Exception as e:
                print(f"处理图片 {filename} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()

    # 关闭所有窗口
    cv2.destroyAllWindows()


def main():
    # 初始化预测器
    model_path = "./detector/weights.28-3.73.hdf5"  # 修改为您的模型路径

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 - {model_path}")
        print("请从以下链接下载模型: https://github.com/yu4u/age-gender-estimation/releases")
        print("下载后放置在 ./detector/ 目录下")
        return

    print(f"加载模型: {model_path}")

    try:
        predictor = UTKFacePredictor(model_path)
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # 主图片文件夹路径
    main_img_folder = "./img"

    # 检查图片目录是否存在
    if not os.path.exists(main_img_folder):
        print(f"错误: 图片目录不存在 - {main_img_folder}")
        return

    # 遍历主图片文件夹中的所有子文件夹
    for folder_name in sorted(os.listdir(main_img_folder)):
        folder_path = os.path.join(main_img_folder, folder_name)

        # 确保是文件夹
        if os.path.isdir(folder_path):
            print(f"\n{'=' * 50}")
            print(f"处理文件夹: {folder_name}")
            print(f"{'=' * 50}")
            process_images_in_folder(folder_path, predictor)


if __name__ == "__main__":
    main()
