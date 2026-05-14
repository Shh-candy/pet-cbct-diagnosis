# config.py
KNOWLEDGE_BASE_PATH = "knowledge_base.json"
SAMPLE_IMAGES_PATH = os.path.abspath(os.getenv("SAMPLE_IMAGES_PATH", DEFAULT_SAMPLE_PATH))
print(f"最终图片路径：{SAMPLE_IMAGES_PATH}")
