import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import ast
import re
from typing import Dict, List, Tuple
import warnings
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# 這支模型支援中英文，768 維；也可換更小的 all-MiniLM-L6-v2 (384 維)


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


def train_model(model, train_loader, criterion, optimizer, epochs=10000):
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
    
def main():
    rec_system = HybridRecommendationSystem()
    rec_system.load_data('課程資料_20250604_215356.csv', '模擬學生總結資料.csv')
    rec_system.preprocess_data()

    n_students = len(rec_system.student_encoder.classes_)
    n_courses  = len(rec_system.course_encoder.classes_)

    cf_model = CollaborativeFilteringModel(n_students, n_courses)
    cb_model = ContentBasedModel(
        rec_system.student_features.shape[1],
        rec_system.course_content_features.shape[1]
    )
    hybrid_model = HybridModel(cf_model, cb_model)

    # === 建 DataLoader ===
    train_dataset = InteractionDataset(rec_system)
    train_loader  = DataLoader(train_dataset, batch_size=512, shuffle=True)

    # === 訓練 ===
    criterion  = nn.BCELoss()
    optimizer  = optim.Adam(hybrid_model.parameters(), lr=1e-3)
    train_model(hybrid_model, train_loader, criterion, optimizer, epochs=5)

    # === 假裝第一筆學生就是「你」 ===
    first_student_id = rec_system.student_data.iloc[0]['student_id']
    top_recs = get_recommendations(hybrid_model, first_student_id, rec_system, top_k=10)

    print(f"\n📝 針對學生 {first_student_id} 推薦課程：")
    for rank, (code, score) in enumerate(top_recs, 1):
        name = rec_system.course_data.loc[
            rec_system.course_data['課程編碼'] == code, '課程名稱'
        ].values[0]
        print(f"{rank:2d}. {code} {name:<20} ▶ 預測分數 {score:.3f}")

    return rec_system, hybrid_model

if __name__ == "__main__":
    rec_system, model = main()

