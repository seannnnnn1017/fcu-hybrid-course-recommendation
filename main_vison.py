import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

import ast
import re
from typing import Dict, List, Tuple
import warnings
from torch.utils.data import Dataset, DataLoader
import streamlit as st
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

warnings.filterwarnings('ignore')

class InteractionDataset(Dataset):
    """把互動紀錄轉成 PyTorch Dataset"""

    def __init__(self, rec_system):
        df = rec_system.interactions_df
        self.student_ids  = torch.LongTensor(df['student_idx'].values)
        self.course_ids   = torch.LongTensor(df['course_idx'].values)
        self.ratings      = torch.FloatTensor(df['rating'].values)

        # 對齊對應的內容特徵
        self.student_features = torch.FloatTensor(
            rec_system.student_features[self.student_ids]
        )
        self.course_features = torch.FloatTensor(
            rec_system.course_content_features[self.course_ids]
        )

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (self.student_ids[idx],
                self.course_ids[idx],
                self.student_features[idx],
                self.course_features[idx],
                self.ratings[idx])


class HybridRecommendationSystem:
    """
    基於深度學習的混合推薦系統
    結合協同過濾(Collaborative Filtering)和內容過濾(Content-based Filtering)
    """
    
    def __init__(self, embedding_dim=64, hidden_dims=[128, 64, 32]):
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 數據存儲
        self.course_data = None
        self.student_data = None
        self.interaction_matrix = None
        
        # 特徵編碼器
        self.student_encoder = LabelEncoder()
        self.course_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.scaler = StandardScaler()
        
        # 模型組件
        self.collaborative_model = None
        self.content_model = None
        self.hybrid_model = None
        
    def load_data(self, course_file, student_file):
        """載入課程和學生數據"""
        self.course_data = pd.read_csv(course_file)
        self.student_data = pd.read_csv(student_file)
        
        # 處理學生已修課程列表
        self.student_data['passed_courses'] = self.student_data['passed_courses'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        
        print(f"載入課程數據: {len(self.course_data)} 門課程")
        print(f"載入學生數據: {len(self.student_data)} 名學生")
        
    def preprocess_data(self):
        """數據預處理"""
        # 1. 創建交互矩陣 (學生-課程)
        self._create_interaction_matrix()
        
        # 2. 處理課程內容特徵
        self._process_course_features()
        
        # 3. 處理學生特徵
        self._process_student_features()
        
        print("數據預處理完成")
        
    def _create_interaction_matrix(self):
        """創建學生-課程交互矩陣"""
        # 展開學生課程記錄
        interactions = []
        for _, student in self.student_data.iterrows():
            for course_code in student['passed_courses']:
                interactions.append({
                    'student_id': student['student_id'],
                    'course_code': course_code,
                    'rating': 1.0  # 假設已修課程為正向反饋
                })
        
        self.interactions_df = pd.DataFrame(interactions)
        
        # 編碼學生和課程ID
        self.interactions_df['student_idx'] = self.student_encoder.fit_transform(
            self.interactions_df['student_id']
        )
        self.interactions_df['course_idx'] = self.course_encoder.fit_transform(
            self.interactions_df['course_code']
        )
        
        # 創建交互矩陣
        n_students = len(self.student_encoder.classes_)
        n_courses = len(self.course_encoder.classes_)
        
        self.interaction_matrix = np.zeros((n_students, n_courses))
        for _, row in self.interactions_df.iterrows():
            self.interaction_matrix[int(row['student_idx']), int(row['course_idx'])] = row['rating']
            
    def _process_course_features(self):
        # 1. 文字特徵 ↦ SBERT
        texts = (
            self.course_data['課程名稱'].fillna('') + ' ' +
            self.course_data['課程描述'].fillna('')
        ).tolist()                         # list[str]
        with torch.no_grad():
            embeddings = sbert_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            # embeddings.shape = (n_courses, 768)

        # 2. 數值特徵照舊
        num_feats = []
        for _, c in self.course_data.iterrows():
            num_feats.append([
                c['學分'],
                1 if c['必選修'] == '必修' else 0,
                1 if c['上課方式'] == '課堂教學' else 0
            ])
        num_feats = self.scaler.fit_transform(np.array(num_feats))

        # 3. 串接 (或分開餵模型都可)
        self.course_content_features = np.hstack([embeddings, num_feats])
        
    def _process_student_features(self):
        """處理學生特徵"""
        student_features = []
        for _, student in self.student_data.iterrows():
            features = [
                student['grade_level'],
                student['gpa'],
                student['total_credits'],
                len(student['passed_courses'])
            ]
            student_features.append(features)
            
        self.student_features = np.array(student_features)
        self.student_features = self.scaler.fit_transform(self.student_features)


class CollaborativeFilteringModel(nn.Module):
    """協同過濾神經網絡模型"""
    
    def __init__(self, n_students, n_courses, embedding_dim=64, hidden_dims=[128, 64]):
        super(CollaborativeFilteringModel, self).__init__()
        
        # 嵌入層
        self.student_embedding = nn.Embedding(n_students, embedding_dim)
        self.course_embedding = nn.Embedding(n_courses, embedding_dim)
        
        # 全連接層
        layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, student_ids, course_ids):
        student_emb = self.student_embedding(student_ids)
        course_emb = self.course_embedding(course_ids)
        
        # 連接嵌入向量
        x = torch.cat([student_emb, course_emb], dim=1)
        output = self.mlp(x)
        
        return output.squeeze(1)


class ContentBasedModel(nn.Module):
    """基於內容的神經網絡模型"""
    
    def __init__(self, student_feature_dim, course_feature_dim, hidden_dims=[128, 64]):
        super(ContentBasedModel, self).__init__()
        
        # 學生特徵編碼器
        self.student_encoder = nn.Sequential(
            nn.Linear(student_feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 課程特徵編碼器
        self.course_encoder = nn.Sequential(
            nn.Linear(course_feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 相似度計算
        layers = []
        input_dim = 64  # 32 + 32
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, student_features, course_features):
        student_encoded = self.student_encoder(student_features)
        course_encoded = self.course_encoder(course_features)
        
        # 連接特徵
        x = torch.cat([student_encoded, course_encoded], dim=1)
        output = self.mlp(x)
        
        return output.squeeze(1)


class HybridModel(nn.Module):
    """混合推薦模型"""
    
    def __init__(self, collaborative_model, content_model, fusion_dim=32):
        super(HybridModel, self).__init__()
        
        self.collaborative_model = collaborative_model
        self.content_model = content_model
        
        # 融合層
        self.fusion_layer = nn.Sequential(
            nn.Linear(2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid()
        )
        
        # 可學習的權重
        self.cf_weight = nn.Parameter(torch.tensor(0.5))
        self.cb_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, student_ids, course_ids, student_features, course_features):
        # 協同過濾預測
        cf_output = self.collaborative_model(student_ids, course_ids)
        
        # 內容過濾預測
        cb_output = self.content_model(student_features, course_features)
        
        # 加權融合
        weights = torch.softmax(torch.stack([self.cf_weight, self.cb_weight]), dim=0)
        weighted_output = weights[0] * cf_output + weights[1] * cb_output
        
        # 進一步融合
        fusion_input = torch.stack([cf_output, cb_output], dim=1)
        final_output = self.fusion_layer(fusion_input)
        
        # 組合最終輸出
        return 0.7 * weighted_output + 0.3 * final_output.squeeze(1)


def train_model(model, train_loader, criterion, optimizer, epochs=100):
    """訓練模型"""
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            student_ids, course_ids, student_features, course_features, ratings = batch
            
            predictions = model(student_ids, course_ids, student_features, course_features)
            loss = criterion(predictions, ratings.float())
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
            
    return losses


def get_recommendations(model, student_id, rec_sys, top_k=10):
    model.eval()
    with torch.no_grad():
        # 1. 取得學生索引與特徵
        if student_id not in rec_sys.student_encoder.classes_:
            return []

        sid = rec_sys.student_encoder.transform([student_id])[0]
        stu_feat = torch.FloatTensor(rec_sys.student_features[sid:sid+1])

        # 2. 已修課與「必修」課程清單
        taken = set(rec_sys.student_data.loc[
            rec_sys.student_data['student_id'] == student_id, 'passed_courses'
        ].iloc[0])

        # 先把必修課程編碼成 set，加速查詢
        required_set = set(
            rec_sys.course_data.loc[rec_sys.course_data['必選修'] == '必修', '課程編碼']
        )

        recs = []
        for cidx, ccode in enumerate(rec_sys.course_encoder.classes_):
            # 過濾：已修課 or 必修
            if ccode in taken or ccode in required_set:
                continue

            course_feat = torch.FloatTensor(
                rec_sys.course_content_features[cidx:cidx+1]
            )
            score = model(
                torch.LongTensor([sid]),
                torch.LongTensor([cidx]),
                stu_feat,
                course_feat
            )
            recs.append((ccode, float(score)))

        # 3. 依分數排序取 Top-k
        recs.sort(key=lambda x: x[1], reverse=True)
        return recs[:top_k]

def initialize_system():
    """初始化推薦系統"""
    if 'rec_system' not in st.session_state:
        st.session_state['rec_system'] = None
        st.session_state['hybrid_model'] = None
        st.session_state['model_trained'] = False

def main():
    st.set_page_config(
        page_title="🎓 課程推薦系統",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 初始化系統
    initialize_system()
    
    # 主標題
    st.title("🎓 智能課程推薦系統")
    st.markdown("---")
    
    # 側邊欄
    with st.sidebar:
        st.header("⚙️ 系統設定")
        
        # 檔案上傳
        st.subheader("📁 資料檔案")
        course_file = st.file_uploader(
            "上傳課程資料CSV", 
            type=['csv'],
            key="course_file"
        )
        student_file = st.file_uploader(
            "上傳學生資料CSV", 
            type=['csv'],
            key="student_file"
        )
        
        # 推薦參數
        st.subheader("📊 推薦設定")
        top_k = st.slider("推薦課程數量", 5, 20, 10)
        
    # 主要內容區域
    if course_file is not None and student_file is not None:
        # 載入資料
        with st.spinner("載入資料中..."):
            course_data = pd.read_csv(course_file)
            student_data = pd.read_csv(student_file)
        
        # 顯示資料概覽
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📚 課程資料概覽")
            st.dataframe(course_data.head(), use_container_width=True)
            st.info(f"總共 {len(course_data)} 門課程")
            
        with col2:
            st.subheader("👥 學生資料概覽")
            st.dataframe(student_data.head(), use_container_width=True)
            st.info(f"總共 {len(student_data)} 位學生")
        
        # 資料視覺化
        st.subheader("📈 資料分析")
        
        # 創建標籤頁
        tab2,tab3 = st.tabs(["推薦系統","互動分析"])
        
        with tab3:
            st.subheader("🔗 學生課程互動分析")
            
            if 'passed_courses' in student_data.columns:
                # 課程熱門度分析
                col1, col2 = st.columns(2)
                
                with col1:
                    try:
                        import ast
                        from collections import Counter
                        
                        # 統計所有課程的選修次數
                        all_courses = []
                        for courses_str in student_data['passed_courses'].dropna():
                            try:
                                courses = ast.literal_eval(courses_str)
                                all_courses.extend(courses)
                            except:
                                continue
                        
                        course_popularity = Counter(all_courses)
                        top_courses = course_popularity.most_common(15)
                        
                        if top_courses:
                            courses_df = pd.DataFrame(top_courses, columns=['課程代碼', '選修人數'])
                            
                            fig = px.bar(
                                courses_df,
                                x='選修人數',
                                y='課程代碼',
                                orientation='h',
                                title="熱門課程排行榜 (前15名)",
                                color='選修人數',
                                color_continuous_scale='plasma'
                            )
                            fig.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"課程熱門度分析錯誤: {str(e)}")
                
                with col2:
                    # GPA vs 修課數量散點圖
                    try:
                        student_analysis = student_data.copy()
                        student_analysis['course_count'] = student_analysis['passed_courses'].apply(
                            lambda x: len(ast.literal_eval(x)) if pd.notna(x) and isinstance(x, str) and x.startswith('[') else 0
                        )
                        
                        fig = px.scatter(
                            student_analysis,
                            x='course_count',
                            y='gpa',
                            size='total_credits',
                            color='grade_level',
                            title="GPA vs 修課數量關係",
                            hover_data=['student_id']
                        )
                        fig.update_layout(
                            xaxis_title="已修課程數量",
                            yaxis_title="GPA"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"散點圖分析錯誤: {str(e)}")
                
                # 年級課程選修模式
                st.subheader("📈 各年級課程選修模式")
                
                try:
                    grade_course_stats = []
                    for grade in sorted(student_data['grade_level'].unique()):
                        grade_students = student_data[student_data['grade_level'] == grade]
                        
                        total_courses = 0
                        avg_gpa = grade_students['gpa'].mean()
                        avg_credits = grade_students['total_credits'].mean()
                        
                        for courses_str in grade_students['passed_courses'].dropna():
                            try:
                                courses = ast.literal_eval(courses_str)
                                total_courses += len(courses)
                            except:
                                continue
                        
                        avg_courses = total_courses / len(grade_students) if len(grade_students) > 0 else 0
                        
                        grade_course_stats.append({
                            '年級': f"{grade}年級",
                            '平均GPA': avg_gpa,
                            '平均學分': avg_credits,
                            '平均修課數': avg_courses,
                            '學生人數': len(grade_students)
                        })
                    
                    grade_stats_df = pd.DataFrame(grade_course_stats)
                    
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        fig = px.line(
                            grade_stats_df,
                            x='年級',
                            y='平均GPA',
                            title="各年級平均GPA趨勢",
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col4:
                        fig = px.bar(
                            grade_stats_df,
                            x='年級',
                            y='平均修課數',
                            title="各年級平均修課數",
                            color='平均修課數',
                            color_continuous_scale='viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 顯示統計表格
                    st.write("### 📋 各年級統計摘要")
                    st.dataframe(grade_stats_df, use_container_width=True, hide_index=True)
                    
                except Exception as e:
                    st.error(f"年級分析錯誤: {str(e)}")
            
            else:
                st.info("缺少課程互動資料，無法進行分析")
        

        with tab2:
            # 模型訓練區域
            st.subheader("🤖 模型訓練")
            
            if st.button("🚀 開始訓練模型", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # 保存上傳的文件到臨時文件
                    import tempfile
                    import os
                    
                    with tempfile.NamedTemporaryFile(mode='w+b', suffix='.csv', delete=False) as tmp_course:
                        tmp_course.write(course_file.getvalue())
                        course_temp_path = tmp_course.name
                    
                    with tempfile.NamedTemporaryFile(mode='w+b', suffix='.csv', delete=False) as tmp_student:
                        tmp_student.write(student_file.getvalue())
                        student_temp_path = tmp_student.name
                    
                    status_text.text("初始化推薦系統...")
                    progress_bar.progress(20)
                    
                    rec_system = HybridRecommendationSystem()
                    rec_system.load_data(course_temp_path, student_temp_path)
                    rec_system.preprocess_data()

                    progress_bar.progress(40)
                    status_text.text("建構模型...")
                    
                    n_students = len(rec_system.student_encoder.classes_)
                    n_courses  = len(rec_system.course_encoder.classes_)

                    cf_model = CollaborativeFilteringModel(n_students, n_courses)
                    cb_model = ContentBasedModel(
                        rec_system.student_features.shape[1],
                        rec_system.course_content_features.shape[1]
                    )
                    hybrid_model = HybridModel(cf_model, cb_model)

                    progress_bar.progress(60)
                    status_text.text("準備訓練數據...")
                    
                    # === 建 DataLoader ===
                    train_dataset = InteractionDataset(rec_system)
                    train_loader  = DataLoader(train_dataset, batch_size=512, shuffle=True)

                    progress_bar.progress(70)
                    status_text.text("訓練模型...")
                    
                    # === 訓練 ===
                    criterion  = nn.BCELoss()
                    optimizer  = optim.Adam(hybrid_model.parameters(), lr=1e-3)
                    train_model(hybrid_model, train_loader, criterion, optimizer, epochs=5)
                    
                    progress_bar.progress(90)
                    status_text.text("保存模型...")
                    
                    # 保存到 session state
                    st.session_state['rec_system'] = rec_system
                    st.session_state['hybrid_model'] = hybrid_model
                    st.session_state['model_trained'] = True
                    
                    progress_bar.progress(100)
                    status_text.text("訓練完成！")
                    
                    st.success("✅ 模型訓練成功！")
                    
                    # 清理臨時文件
                    os.unlink(course_temp_path)
                    os.unlink(student_temp_path)
                    
                except Exception as e:
                    st.error(f"❌ 訓練過程中發生錯誤: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
            
            # 推薦系統界面
            if st.session_state.get('model_trained', False):
                st.subheader("🎯 課程推薦")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # 學生選擇
                    if 'student_id' in student_data.columns:
                        # 創建更詳細的學生選項顯示
                        student_options = []
                        for _, row in student_data.iterrows():
                            student_info = f"{row['student_id']} (年級: {row.get('grade_level', 'N/A')}, GPA: {row.get('gpa', 'N/A'):.2f})"
                            student_options.append((row['student_id'], student_info))
                        
                        selected_student = st.selectbox(
                            "選擇學生",
                            options=[opt[0] for opt in student_options],
                            format_func=lambda x: next(opt[1] for opt in student_options if opt[0] == x),
                            key="student_selector"
                        )
                        
                        # 顯示選中學生的詳細資訊
                        if selected_student:
                            student_info = student_data[student_data['student_id'] == selected_student].iloc[0]
                            
                            st.write("### 👤 學生資訊")
                            info_col1, info_col2 = st.columns(2)
                            
                            with info_col1:
                                st.write(f"**學號**: {student_info['student_id']}")
                                st.write(f"**年級**: {student_info.get('grade_level', 'N/A')}")
                            
                            with info_col2:
                                st.write(f"**GPA**: {student_info.get('gpa', 'N/A'):.2f}")
                                st.write(f"**總學分**: {student_info.get('total_credits', 'N/A')}")
                            
                            # 顯示已修課程
                            if 'passed_courses' in student_info and pd.notna(student_info['passed_courses']):
                                try:
                                    import ast
                                    passed_courses = ast.literal_eval(student_info['passed_courses'])
                                    st.write(f"**已修課程** ({len(passed_courses)} 門):")
                                    
                                    # 以標籤形式顯示課程
                                    courses_html = ""
                                    for course in passed_courses:
                                        courses_html += f'<span style="background-color: #e1f5fe; padding: 2px 8px; margin: 2px; border-radius: 12px; font-size: 12px; display: inline-block;">{course}</span> '
                                    
                                    st.markdown(courses_html, unsafe_allow_html=True)
                                except:
                                    st.write("**已修課程**: 資料解析錯誤")
                    
                    if st.button("🔍 生成推薦", type="secondary"):
                        with st.spinner("生成推薦中..."):
                            # 使用真實的推薦邏輯
                            rec_system = st.session_state['rec_system']
                            hybrid_model = st.session_state['hybrid_model']
                            
                            top_recs = get_recommendations(hybrid_model, selected_student, rec_system, top_k=top_k)
                            
                            # 將課程代碼轉換為完整的課程資訊
                            recommendations = []
                            for course_code, score in top_recs:
                                # 查找課程名稱
                                course_info = course_data[course_data['課程編碼'] == course_code]
                                if not course_info.empty:
                                    course_name = course_info.iloc[0]['課程名稱']
                                else:
                                    course_name = f"課程 {course_code}"
                                
                                recommendations.append((course_code, course_name, score))
                            
                            st.session_state['recommendations'] = recommendations
                
        
            # 顯示推薦結果
            if 'recommendations' in st.session_state and st.session_state['recommendations']:
                st.write("### 📋 推薦結果")
                recommendations = st.session_state['recommendations']
                
                # 創建推薦課程的視覺化卡片
                for i, (course_code, course_name, score) in enumerate(recommendations, 1):
                    # 查找完整課程資訊
                    course_detail = course_data[course_data['課程編碼'] == course_code]
                    
                    if not course_detail.empty:
                        course_row = course_detail.iloc[0]
                        
                        # 計算推薦信心度
                        confidence = "高" if score > 0.7 else "中" if score > 0.5 else "低"
                        confidence_color = "#4CAF50" if score > 0.7 else "#FF9800" if score > 0.5 else "#F44336"
                        
                        # 課程卡片
                        with st.container():
                            st.markdown(f"""
                            <div style="
                                border: 1px solid #ddd; 
                                border-radius: 10px; 
                                padding: 15px; 
                                margin: 10px 0; 
                                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            ">
                                <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 10px;">
                                    <h4 style="margin: 0; color: #2c3e50;">#{i} {course_name}</h4>
                                    <span style="
                                        background-color: {confidence_color}; 
                                        color: white; 
                                        padding: 4px 12px; 
                                        border-radius: 20px; 
                                        font-size: 12px; 
                                        font-weight: bold;
                                    ">
                                        信心度: {confidence} ({score:.3f})
                                    </span>
                                </div>
                                <p style="margin: 5px 0;"><strong>課程編碼:</strong> {course_code}</p>
                                <p style="margin: 5px 0;"><strong>學分:</strong> {course_row.get('學分', 'N/A')}</p>
                                <p style="margin: 5px 0;"><strong>必選修:</strong> {course_row.get('必選修', 'N/A')}</p>
                                <p style="margin: 5px 0;"><strong>上課方式:</strong> {course_row.get('上課方式', 'N/A')}</p>
                                <p style="margin: 5px 0;"><strong>課程描述:</strong> {course_row.get('課程描述', '無描述')[:100]}{'...' if len(str(course_row.get('課程描述', ''))) > 100 else ''}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # 如果找不到課程詳細資訊
                        st.info(f"#{i} {course_name} (課程編碼: {course_code}) - 推薦分數: {score:.3f}")
                
                # 推薦結果統計
                st.write("### 📊 推薦統計")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    high_conf = sum(1 for _, _, score in recommendations if score > 0.7)
                    st.metric("高信心度推薦", high_conf)
                
                with col2:
                    avg_score = sum(score for _, _, score in recommendations) / len(recommendations)
                    st.metric("平均推薦分數", f"{avg_score:.3f}")
                
                with col3:
                    st.metric("總推薦課程", len(recommendations))
                
                # 推薦分數分布圖
                if len(recommendations) > 0:
                    scores_df = pd.DataFrame([
                        {'課程': f"{name[:20]}..." if len(name) > 20 else name, 
                            '推薦分數': score} 
                        for _, name, score in recommendations
                    ])
                    
                    fig = px.bar(
                        scores_df, 
                        x='推薦分數', 
                        y='課程',
                        orientation='h',
                        title="推薦分數分布",
                        color='推薦分數',
                        color_continuous_scale='plasma'
                    )
                    fig.update_layout(
                        yaxis={'categoryorder':'total ascending'},
                        height=max(400, len(recommendations) * 30)
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("👆 請先選擇學生並點擊「生成推薦」按鈕")
        
    else:
        # 系統說明
        st.info("""
        ### 📝 使用說明
        
        1. **上傳資料檔案**: 請在左側邊欄上傳課程資料CSV和學生資料CSV
        2. **查看資料概覽**: 系統會自動載入並顯示資料統計
        3. **訓練模型**: 點擊「開始訓練模型」按鈕來訓練混合推薦系統
        4. **生成推薦**: 選擇學生並獲得個人化課程推薦
        
        ### 📋 資料格式要求
        
        **課程資料CSV應包含以下欄位:**
        - 課程編碼, 課程名稱, 課程描述, 學分, 必選修, 上課方式
        
        **學生資料CSV應包含以下欄位:**
        - student_id, grade_level, gpa, total_credits, passed_courses
        
        其中 `passed_courses` 應為列表格式，例如: `['CS101', 'MATH201', 'ENG301']`
        """)
        
        # 系統架構說明
        with st.expander("🔧 系統架構說明"):
            st.markdown("""
            ### 混合推薦系統架構
            
            本系統結合了三種推薦方法：
            
            1. **協同過濾 (Collaborative Filtering)**
               - 基於用戶-物品互動矩陣
               - 使用神經網絡學習用戶和課程的嵌入向量
               
            2. **內容過濾 (Content-based Filtering)**  
               - 基於課程內容特徵和學生特徵
               - 使用SBERT處理文本特徵
               
            3. **混合模型 (Hybrid Model)**
               - 融合協同過濾和內容過濾的預測結果
               - 使用可學習權重進行自適應融合
            
            ### 特色功能
            - 🎯 個人化推薦
            - 📊 互動式資料視覺化  
            - 🤖 深度學習模型
            - 📈 推薦結果分析
            """)


if __name__ == "__main__":
    main()