import cv2
import numpy as np
from PIL import Image
import os
import base64
import io
# 假设你的config.py中定义了SAMPLE_IMAGES_PATH，若未定义可手动指定
# 示例：SAMPLE_IMAGES_PATH = r"C:\Users\shh\Desktop\diagnosis_agent\data\sample_images"

class CBCTImageProcessor:
    def __init__(self):
        # 支持CBCT常用格式：常规图片+DICOM原生格式
        self.image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".dcm"]

    def load_image(self, image_name):
        """加载CBCT影像（支持常规格式+DICOM），返回PIL.Image对象"""
        # 1. 拼接完整路径并校验文件存在性
        if not isinstance(image_name, str):
            raise TypeError(f"图片名必须是字符串，当前类型：{type(image_name)}")
        image_path = os.path.join(SAMPLE_IMAGES_PATH, image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"CBCT影像文件不存在：{image_path}")
        
        # 2. 校验文件格式
        ext = os.path.splitext(image_name)[1].lower()
        if ext not in self.image_extensions:
            raise ValueError(f"不支持的影像格式：{ext}，仅支持{self.image_extensions}")

        # 3. 处理DICOM格式（CBCT原生格式，优先支持）
        if ext == ".dcm":
            try:
                import pydicom
                from pydicom.errors import InvalidDicomError

                ds = pydicom.dcmread(image_path)
                # 提取DICOM像素数据并归一化到0-255
                img_gray = ds.pixel_array
                if img_gray.dtype != np.uint8:
                    min_val = np.min(img_gray)
                    max_val = np.max(img_gray)
                    # 防止除以0（边界保护）
                    if max_val - min_val == 0:
                        img_gray = np.zeros_like(img_gray, dtype=np.uint8)
                    else:
                        img_gray = (img_gray - min_val) / (max_val - min_val) * 255
                        img_gray = img_gray.astype(np.uint8)
                # 转为RGB（多模态模型通用输入格式）
                img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
                image = Image.fromarray(img_rgb)

            except ImportError:
                raise ImportError("请安装pydicom库支持DICOM格式：pip install pydicom")
            except InvalidDicomError:
                raise ValueError(f"{image_path} 不是有效的DICOM文件")
            except Exception as e:
                raise RuntimeError(f"DICOM文件处理失败：{str(e)}")
        
        # 4. 处理常规图片格式
        else:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图片：{image_path}（文件损坏/格式不支持）")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img_rgb)

        # 5. 返回处理后的PIL.Image对象（核心：load_image必须有返回值）
        return image
    

    def batch_load_images(self, image_names):
        """批量加载多张图片（支持常规格式+DICOM）"""
        if not isinstance(image_names, list):
            raise TypeError(f"批量加载需传入列表，当前类型：{type(image_names)}")
        
        batch_images = []
        for img_name in image_names:
            try:
                img = self.load_image(img_name)
                batch_images.append(img)
                print(f"成功加载：{img_name}，分辨率：{img.size[0]}×{img.size[1]}")
            except Exception as e:
                raise ValueError(f"加载图片{img_name}失败：{e}")
        return batch_images

    def batch_image_to_base64(self, batch_images):
        """批量转换多张图片为Base64"""
        batch_base64 = []
        for idx, img in enumerate(batch_images):
            img_b64 = self.image_to_base64(img)
            batch_base64.append(img_b64)
            print(f"完成图片{idx+1}的Base64编码（前50位）：{img_b64[:50]}...")
        return batch_base64

    def image_to_base64(self, image):
        """将PIL.Image转为Base64编码（类方法，修正self参数）"""
        buffer = io.BytesIO()
        # 保存为PNG格式（无损，适配多模态模型）
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return img_base64

# 统一的测试入口（合并重复的if __name__）
if __name__ == "__main__":
    processor = CBCTImageProcessor()
    try:
        # 测试1：单张图片加载+Base64转换
        print("===== 测试单张图片处理 =====")
        img = processor.load_image("sample_gushi.png")  # 替换为你的图片名
        img_b64 = processor.image_to_base64(img)
        print(f"影像加载成功，分辨率：{img.size[0]}×{img.size[1]}")
        print(f"Base64编码前100位：{img_b64[:100]}...")

        # 测试2：批量图片处理
        print("\n===== 测试批量图片处理 =====")
        img_names = ["sample_gushi.png", "test_lung.png"]  # 替换为你的图片名
        batch_imgs = processor.batch_load_images(img_names)
        batch_b64 = processor.batch_image_to_base64(batch_imgs)
        print(f"\n批量处理完成，共加载{len(batch_imgs)}张图片")

    except Exception as e:
        print(f"处理失败：{e}")
