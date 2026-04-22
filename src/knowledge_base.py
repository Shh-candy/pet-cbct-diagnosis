import os
import json
from config import KNOWLEDGE_BASE_PATH

class PetCBCTKnowledgeBase:
    def __init__(self):
        # 只保留 Vet-Anatomy 标准解剖知识库
        self.vet_anatomy_file = os.path.join(KNOWLEDGE_BASE_PATH, "vet_anatomy_ct.json")
        self.vet_data = self.load_vet_anatomy()
        print("✅ 宠物CBCT解剖知识库（Vet-Anatomy）加载完成")

    def load_vet_anatomy(self):
        """仅加载解剖标准，不加载任何疾病知识"""
        if os.path.exists(self.vet_anatomy_file):
            with open(self.vet_anatomy_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get_normal_reference(self, species: str, part: str):
        """获取该物种某个部位的正常参考值"""
        species_data = self.vet_data.get("species", {}).get(species, {})
        return species_data.get("normal_anatomy", {}).get(part, {})

    def get_standard_views(self, species):
        """标准切面"""
        return self.vet_data.get("species", {}).get(species, {}).get("standard_views", {})

    def get_artifact_guide(self):
        """伪影鉴别"""
        return self.vet_data.get("artifact_guide", {})

    def get_knowledge_context(self, pet_info):
        """
        构建给LLM的解剖知识库上下文(无任何疾病内容)
        """
        # 判断是犬还是猫
        species = "犬" if any(x in pet_info for x in ["犬", "狗", "金毛", "边牧"]) else "猫"

        # 正常参考
        jaw = self.get_normal_reference(species, "颌骨")
        tooth = self.get_normal_reference(species, "牙齿")
        sinus = self.get_normal_reference(species, "鼻窦")
        views = self.get_standard_views(species)
        artifact = self.get_artifact_guide()

        # 最终给模型的知识库
        return f"""
【Vet-Anatomy 标准解剖参考（{species}）】
1. 正常参考值
• 颌骨：{jaw.get("标准灰度范围", "无数据")}，形态：{jaw.get("形态特征", "无")}
• 牙齿：{tooth.get("标准灰度范围", "无数据")}，形态：{tooth.get("形态特征", "无")}
• 鼻窦：{sinus.get("标准灰度范围", "无数据")}，形态：{sinus.get("形态特征", "无")}

2. 标准切面
• 轴位：{', '.join(views.get("轴位", []))}
• 冠状：{', '.join(views.get("冠状", []))}
• 矢状：{', '.join(views.get("矢状", []))}

3. 伪影鉴别
• 正常伪影：{', '.join(artifact.get("正常伪影", []))}
• 鉴别要点：{artifact.get("鉴别要点", "无")}

【诊断规则】
- 仅根据解剖结构是否正常进行判断
- 不臆断疾病
- 只描述：结构是否连续、边界是否清晰、密度是否正常、位置是否标准
- 使用规范解剖术语
"""