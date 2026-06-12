import streamlit as st
import os


api_key = st.secrets["DOUBAO_API_KEY"]
base_url = st.secrets["DOUBAO_BASE_URL"]
model_name = st.secrets["MODEL_NAME"]


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from cbct_image_processor import CBCTImageProcessor
from langchain_core.messages import HumanMessage, SystemMessage
from knowledge_base import PetCBCTKnowledgeBase


class PetCBCTDiagnosisAgent:
    def __init__(self):
        # 初始化LLM（支持多模态）
       ## TEMPERATURE = 0.05  # 诊断场景低随机性
       # MAX_TOKENS = 2000   # 输出最大长度
       temperature=0.7
       timeout=60

       self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,      # DeepSeek模型名
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
            你是专业的宠物影像诊断兽医团队,具备CBCT(锥形束计算机断层扫描）图像解读能力,擅长解读犬猫全身各部位CBCT影像，基于专业的兽医影像知识为用户（宠物主人或兽医）提供辅助诊断建议。
            以专业术语解释复杂的影像特征，确保用户能够理解关键诊断信息。
            请根据Vet-Anatomy解剖知识库,为该宠物的CBCT影像提供详细的诊断分析:

            ###宠物基本信息
            {pet_info}

             ### 【读片基本规范（必须严格遵守）】
            1.  **先定位再描述**：先明确影像对应的解剖部位（如：头面部、胸腔、腹腔、四肢骨骼等），再按「正常结构→异常发现」的顺序描述。
            2.  **异常描述要标准化**:
                - 位置:具体解剖部位（如:右肾肾盂、子宫、鼓室第4腰椎椎体、右前肢桡骨远端等)
                - 形态：是否扩张、变形、移位、断裂、塌陷
                - 边界：是否清晰、是否毛糙、是否与周围组织分界不清
                - 密度：是高密度影（如骨皮质、结石、钙化灶）、等密度影还是低密度影（如积液、水肿、脂肪、气体）
                - 周围组织：是否受压、移位、浸润、水肿、炎症反应
            3.  **区分正常变异与病理改变**:
                - 正常结构：按解剖学标准描述，注明“未见明显异常”
                - 疑似异常:需注明“疑似XX改变.建议结合临床进一步评估”
                - 明确异常：给出直接描述，如“可见明显扩张/积液/高密度影/骨折线”
            4.  **禁止无依据臆断**：影像中未明确显示的病变，禁止直接下结论，需注明“影像未明确显示，建议结合其他检查确认”
            5.  **鉴别诊断要严谨**：对同一影像表现，列出最可能的几种疾病，并给出鉴别要点（如：高密度影，需区分结石、钙化灶、骨皮质、异物）

            ### 诊断要求（必须严格遵守）
            1. 先给出**初步诊断结论**,明确指出影像中是否存在异常（例如：未见明显异常/疑似XX疾病/高度怀疑XX疾病);
            2. 再分析**异常依据**,描述判断异常区域的位置、形态、密度等关键特征（结合影像特征数值,如灰度均值250提示骨骼钙化异常);
            3. 最后给出**后续建议**（如进一步临床检查、治疗方案、复查周期，符合宠物临床规范）；
            4. 仅基于提供的信息诊断，不臆断，不确定的地方标注“暂无法明确，建议结合临床症状”；
            5.诊断要参考提供的宠物信息，关注性别、症状等，但不要过于依赖宠物信息给出的病症部位和性别年龄等，要以兽医的视角将这些辅助信息作为诊断参考，具体诊断要以影像特征为主；
            6. 鉴别诊断与排除说明：
               针对影像中不典型的表现，说明需要鉴别的疾病及排除依据：（例：影像中肾区高密度影，需与肾结石、肾实质钙化、骨伪影鉴别；根据位置及密度特征，不支持典型结石表现）
            7. 报告结构清晰客观，分点说明，语言为宠物临床医学专业用语，避免口语化;
            8. 结尾附上免责声明(重要提示:本分析为CBCT影像的辅助解读,不构成正式医疗诊断。
            最终诊断与治疗方案需由执业兽医结合病理结果、临床检查及其他检验全面评估后确定。宠物健康问题请及时咨询专业兽医。)


            ### 诊断结果：
            """,
            input_variables=["pet_info", "image_features"]
        )
        # 构建LangChain诊断链（端到端：输入→Prompt→LLM→输出解析）
       self.diagnosis_chain = self.diagnosis_prompt | self.llm | StrOutputParser()

    def diagnose(self, image_names, pet_info="未提供"):
        """
        支持多张图片批量诊断
        :param file_path_list: 图片完整路径列表
        :param pet_info: 宠物信息
        :return: 诊断报告
        """
    try:
            # 批量加载图片
            img_batch = self.image_processor.batch_load_images(file_path_list)
            # 批量转base64
            b64_batch = self.image_processor.batch_image_to_base64(img_batch)

            # 拼接提示文本 + 多张图片多模态内容
            text_content = self.diagnosis_prompt.format(pet_info=pet_info)
            content_list = [
                {
                    "type": "text",
                    "text": text_content
                }
            ]

            # 循环追加所有图片
            for b64_str in b64_batch:
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_str}"
                    }
                })

            # 组装请求体
            multimodal_input = [
                {
                    "role": "user",
                    "content": content_list
                }
            ]

            # 调用模型并解析结果
            response = self.llm.invoke(multimodal_input)
            diagnosis_report = StrOutputParser().invoke(response)

            return diagnosis_report

        except Exception as e:
            return f"【诊断失败】错误原因：{str(e)}\n请检查：1.影像文件名/格式是否正确；2.API密钥是否有效；3.模型名是否正确。"
           

# 测试代码
if __name__ == "__main__":
    agent = PetCBCTDiagnosisAgent()
    # 测试诊断（替换为你的测试图片）
    # 路径放在 [] 内，构成列表
    path_list = [
        r"C:/Users/shh/Desktop/diagnosis_agent/sample_gushi.png",
        r"C:/Users/shh/Desktop/diagnosis_agent/test_lung.png"
        ]
    result = agent.diagnose(path_list, pet_info="金渐层猫,7岁,近期出现呼吸困难")
    print("=== 宠物CBCT影像诊断结果 ===")
    print(result)

