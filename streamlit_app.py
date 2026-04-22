import streamlit as st
import os
import sys
import tempfile
from PIL import Image
# 【必须加在最顶部！】把项目根目录加入 Python 路径
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# 自动导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.agent import PetCBCTDiagnosisAgent

# ====================== 页面配置 ======================
st.set_page_config(
    page_title="宠物CBCT智能诊断",
    page_icon="🐾",
    layout="wide"
)

# ====================== 初始化智能体 ======================
@st.cache_resource
def init_agent():
    try:
        return PetCBCTDiagnosisAgent()
    except Exception as e:
        st.error(f"初始化失败：{str(e)}")
        st.stop()

agent = init_agent()

# ====================== 标题 ======================
st.title("🐾 宠物CBCT智能诊断系统（内部测试版）")
st.divider()

# ====================== 左右分栏布局 ======================
col_left, col_right = st.columns([1, 1.2], gap="large")

# ---------------------- 左侧：上传 + 预览 ----------------------
with col_left:
    st.subheader("📸 影像上传与预览")
    
    uploaded_file = st.file_uploader(
        "选择单张CBCT影像",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=False
    )
    
    pet_info = st.text_area(
        "宠物信息",
        placeholder="例：金渐层，7岁，呼吸困难",
        height=100
    )
    
    run_btn = st.button(
        "开始诊断",
        type="primary",
        use_container_width=True,
        disabled=not (uploaded_file and pet_info)
    )

    # 小图预览（无废弃参数，修复宽度警告）
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.caption("✅ 影像预览")
        st.image(img, width=280)  #  用 width，无废弃参数

# ---------------------- 右侧：诊断报告 ----------------------
with col_right:
    st.subheader("🩺 AI 诊断报告")
    report_placeholder = st.empty()

    if run_btn and uploaded_file:
        with st.spinner("🔍 正在分析影像..."):
            try:
                # 临时文件（无路径限制）
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 执行诊断
                report = agent.diagnose(uploaded_file.name, pet_info)

                # 显示报告：修复 ~~ 删除线问题
                with report_placeholder.container():
                    st.success("✅ 诊断完成")
                    st.divider()
                    # 关键：用 text 而不是 write，避免 markdown 误解析 ~~
                    st.text(report)  

            except Exception as e:
                st.error(f"诊断失败：{str(e)}")
