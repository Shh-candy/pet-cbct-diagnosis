import cv2
import numpy as np
from PIL import Image
import os
from  config import SAMPLE_IMAGES_PATH

class CBCTImageProcessor:
    def __init__(self):
        self.image_extensions = [".png", ".jpg", ".jpeg", ".dcm"]  # 支持DICOM（CBCT常用格式）

    def load_image(self, image_name):
        """加载CBCT影像，支持常规格式和DICOM（简化版）"""
        image_path = os.path.join(SAMPLE_IMAGES_PATH, image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"影像文件不存在：{image_path}")
        
        # 简单区分格式（实际DICOM需用pydicom库）
        ext = os.path.splitext(image_name)[1].lower()
        if ext == ".dcm":
            # 简化处理：实际需用pydicom.dcmread读取像素数据
            raise NotImplementedError("DICOM格式需安装pydicom库，示例暂支持常规图片")
        else:
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img_rgb)

    def extract_features(self, image):
        """提取CBCT影像关键特征（简化版：边缘检测+纹理分析）"""
        # 转为灰度图
        img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        # 边缘检测（识别骨骼/病灶轮廓）
        edges = cv2.Canny(img_gray, 50, 150)
        # 计算纹理特征（灰度共生矩阵，简化版）
        mean_gray = np.mean(img_gray)
        std_gray = np.std(img_gray)
        
        # 特征描述（供LLM分析）
        feature_desc = f"""
        CBCT影像特征提取结果：
        1. 灰度均值：{mean_gray:.2f}（反映整体密度，异常值可能提示病灶）
        2. 灰度标准差：{std_gray:.2f}（反映纹理均匀性，异常提示结构不规则）
        3. 边缘检测：检测到{np.sum(edges > 0)}个边缘像素（骨骼轮廓/病灶边界）
        """
        return feature_desc

# 测试代码
if __name__ == "__main__":
    processor = CBCTImageProcessor()
    try:
        img = processor.load_image("C:/Users/shh/Desktop/diagnosis_agent/sample_gushi.png")  # 替换为你的测试图片
        features = processor.extract_features(img)
        print(features)
    except Exception as e:
        print(f"影像处理出错：{e}")