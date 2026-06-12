import streamlit as st
import os
import sys
import tempfile
from PIL import Image
from pathlib import Path

# 导入
# --------------------------
from agent import PetCBCTDiagnosisAgent

if "image_list" not in st.session_state:
    st.session_state.image_list = []  # 始终保留已上传的所有图片

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
    st.subheader("📸 影像上传")

    # ===== 唯一上传框：允许多选，但靠session实现分次追加 =====
    new_files = st.file_uploader(
        "每次可单选/多选，多次上传会自动追加",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True
    )
    
     # 把本次新选的图片追加到session列表（去重，避免重复上传同一张）
    if new_files:
        existing_names = [f.name for f in st.session_state.image_list]
        for f in new_files:
            if f.name not in existing_names:
                st.session_state.image_list.append(f)
        st.success(f"✅ 本次新增 {len(new_files)} 张，当前共 {len(st.session_state.image_list)} 张")

    # 清空按钮（重置session）
    if st.button("🗑️ 清空所有图片", use_container_width=True):
        st.session_state.image_list = []
        st.experimental_rerun()

    st.divider()

    # 预览所有已上传图片
    st.subheader(" 已上传预览")
    if st.session_state.image_list:
        for idx, img_file in enumerate(st.session_state.image_list):
            try:
                img = Image.open(img_file)
                st.image(img, width=280, caption=f"第 {idx+1} 张：{img_file.name}")
            except:
                st.warning(f"第 {idx+1} 张预览失败")
    else:
        st.info("暂无图片，请上传")

    st.divider()

    # 宠物信息 + 诊断按钮
    pet_info = st.text_area("宠物信息", placeholder="例：金渐层，雌性，7岁，呼吸困难", height=100)
    run_btn = st.button(
        "开始诊断",
        type="primary",
        use_container_width=True,
        disabled=not (st.session_state.image_list and pet_info)
    )

# ---------------------- 右侧：诊断报告 ----------------------
with col_right:
    st.subheader("🩺 AI 诊断报告")
    report_placeholder = st.empty()

    if run_btn and uploaded_file:
        with st.spinner("🔍 正在分析所有影像..."):
            try:
                # 把session里所有图片写到临时目录
                temp_dir = tempfile.mkdtemp()
                file_path_list = []
                for f in st.session_state.image_list:
                    path = os.path.join(temp_dir, f.name)
                    with open(path, "wb") as wf:
                        wf.write(f.getbuffer())
                    file_path_list.append(path)

                # 传给智能体：一次诊断全部图片
                report = agent.diagnose(file_path_list, pet_info)

                # 展示报告
                with report_placeholder.container():
                    st.success("✅ 诊断完成")
                    st.divider()
                    st.text(report)

            except Exception as e:
                st.error(f"诊断失败：{str(e)}")
