import sys
import os

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # 加到项目根目录

api_key = os.getenv("DOUBAO_API_KEY")
base_url = os.getenv("DOUBAO_BASE_URL")
model_name = os.getenv("MODEL_NAME")

BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "src"))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from cbct_image_processor import CBCTImageProcessor
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from knowledge_base import PetCBCTKnowledgeBase

# 加载环境变量
load_dotenv()

class PetCBCTDiagnosisAgent:
    def __init__(self):
        # 初始化LLM（支持多模态）
       ## TEMPERATURE = 0.05  # 诊断场景低随机性
       # MAX_TOKENS = 2000   # 输出最大长度
       temperature=0.7
       timeout=60

       self.llm = ChatOpenAI(
            api_key=DOUBAO_API_KEY,       # 替换为DeepSeek密钥
            base_url=DOUBAO_BASE_URL,     # 替换为DeepSeek API地址
            model_name=MODEL_NAME,      # DeepSeek模型名
            temperature=temperature,
            timeout=timeout
        )

       self.image_processor = CBCTImageProcessor()
       self.kb = PetCBCTKnowledgeBase()  #解剖知识库
       
        # 初始化影像处理器
       self.image_processor = CBCTImageProcessor()
        
        # 诊断提示词模板
       self.diagnosis_prompt = PromptTemplate(
            template="""
            你是专业的宠物影像诊断兽医团队,具备CBCT(锥形束计算机断层扫描）图像解读能力,基于专业的兽医影像知识为用户（宠物主人或兽医）提供辅助诊断建议。
            以专业术语解释复杂的影像特征，确保用户能够理解关键诊断信息。
            请根据Vet-Anatomy解剖知识库,为该宠物的CBCT影像提供详细的诊断分析:

            ### 1. 宠物基本信息
            {pet_info}

            ### 2. 诊断要求（必须严格遵守）
            1. 先给出**初步诊断结论**,明确指出影像中是否存在异常（例如：未见明显异常/疑似XX疾病/高度怀疑XX疾病);
            2. 再分析**异常依据**,描述判断异常区域的位置、形态、密度等关键特征（结合影像特征数值,如灰度均值250提示骨骼钙化异常);
            3. 最后给出**后续建议**（如进一步临床检查、治疗方案、复查周期，符合宠物临床规范）；
            4. 仅基于提供的信息诊断，不臆断，不确定的地方标注“暂无法明确，建议结合临床症状”；
            5. 报告结构清晰客观，分点说明，语言为宠物临床医学专业用语，避免口语化;
            6. 结尾附上免责声明(重要提示:本分析为CBCT影像的辅助解读,不构成正式医疗诊断。
            最终诊断与治疗方案需由执业兽医结合病理结果、临床检查及其他检验全面评估后确定。宠物健康问题请及时咨询专业兽医。)


            ### 诊断结果：
            """,
            input_variables=["pet_info", "image_features"]
        )
        # 构建LangChain诊断链（端到端：输入→Prompt→LLM→输出解析）
       self.diagnosis_chain = self.diagnosis_prompt | self.llm | StrOutputParser()

    def diagnose(self, image_names, pet_info="未提供"):
        """
        执行CBCT影像诊断
        :param image_names: 影像文件名（位于sample_images目录）
        :param pet_info: 宠物基本信息（如品种、年龄、性别、症状）
        :return: 结构化诊断结果（字符串）
        """
        try:
            # 1. 处理影像，提取特征
            image = self.image_processor.load_image(image_names)
            image_b64 = self.image_processor.image_to_base64(image)  # image_features = self.image_processor.extract_features(image)
            # 补充宠物信息 if pet_info:image_features += f"\n### 宠物基本信息：{pet_info}"

            #2：构建多模态输入（DeepSeek标准格式：文本+Base64影像）
            multimodal_input = [
                {
                    "role": "user",
                    "content":[
                  {  "type": "text",
                    "text": self.diagnosis_prompt.format(pet_info=pet_info)
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"  # 固定格式
                    }
                }
            ]
                }
            ]

            # 2. 执行诊断,调用DeepSeek多模态模型直诊，解析输出
            response = self.llm.invoke(multimodal_input)
            diagnosis_report = StrOutputParser().invoke(response)

            return diagnosis_report

        except Exception as e:
            return f"【诊断失败】错误原因：{str(e)}\n请检查：1.影像文件名/格式是否正确；2.API密钥是否有效；3.模型名是否正确。"
            #diagnosis_result = self.diagnosis_chain.invoke({"pet_info": pet_info, "image_features": image_features
           

# 测试代码
if __name__ == "__main__":
    agent = PetCBCTDiagnosisAgent()
    # 测试诊断（替换为你的测试图片）
    result = agent.diagnose(
        image_names="C:/Users/shh/Desktop/diagnosis_agent/sample_gushi.png",
        pet_info="金渐层猫,7岁,近期出现呼吸困难"
    )
    print("=== 宠物CBCT影像诊断结果 ===")
    print(result)

