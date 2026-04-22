import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# DeepSeek API配置（核心修改）
DEEPSEEK_API_KEY = os.getenv("sk-270fcc66191d4e3faaac3ec88df5031d")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# DeepSeek模型参数（适配多模态/文本模型）
LLM_MODEL_NAME = "deepseek-multi-modal"  # DeepSeek通用对话模型（如需多模态用deepseek-multi-modal）
TEMPERATURE = 0.05  # 诊断场景低随机性
MAX_TOKENS = 2000   # 输出最大长度

# OpenAI配置
OPENAI_API_KEY = os.getenv("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")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# 样本图片路径配置
DEFAULT_SAMPLE_PATH = r"C:/Users/shh/Desktop/diagnosis_agent/sample_gushi.png"  # 替换成你实际的路径
SAMPLE_IMAGES_PATH = os.path.abspath(os.getenv("SAMPLE_IMAGES_PATH", DEFAULT_SAMPLE_PATH))
print(f"最终图片路径：{SAMPLE_IMAGES_PATH}")

# 模型参数
LLM_MODEL_NAME = "gpt-4o"  # 支持多模态，可解析图片
TEMPERATURE = 0.1  # 诊断场景需低随机性，保证准确性
CHUNK_SIZE = 1000  # 知识库文档拆分大小
CHUNK_OVERLAP = 200

# 知识库路径配置
# 获取当前文件所在目录（src目录）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 知识库目录：这里设置为 src 下的 knowledge_base_data 文件夹（可根据实际修改）
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # diagnosis_agent 目录
KNOWLEDGE_BASE_PATH = os.path.join(PROJECT_ROOT, "data")