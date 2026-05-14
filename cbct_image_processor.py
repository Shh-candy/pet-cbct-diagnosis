import numpy as np
from PIL import Image
import base64
import io
import pydicom
import os

class CBCTImageProcessor:
    def __init__(self):
        pass

    # 直接处理上传的图片，不依赖任何路径！
    def load_image(self, uploaded_file):
        try:
            # 支持 Streamlit 上传的文件
            if hasattr(uploaded_file, 'name'):
                file_name = uploaded_file.name
                if file_name.endswith('.dcm'):
                    dicom = pydicom.dcmread(uploaded_file)
                    pixel_array = dicom.pixel_array
                    if len(pixel_array.shape) == 3:
                        pixel_array = pixel_array[0]
                    pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255
                    img = Image.fromarray(pixel_array.astype(np.uint8))
                else:
                    img = Image.open(uploaded_file)
                
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return img
            else:
                raise Exception("不支持的文件类型")
        
        except Exception as e:
            raise Exception(f"图片处理失败: {str(e)}")

    def image_to_base64(self, image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
