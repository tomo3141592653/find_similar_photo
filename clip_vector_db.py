import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
import clip
from tqdm import tqdm
import chromadb
from chromadb.config import Settings

class CLIPVectorDB:
    def __init__(self, db_path: str = "./chroma_db", model_name: str = "ViT-B/32"):
        self.db_path = db_path
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        self.init_database()
    
    def init_database(self):
        settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
        self.client = chromadb.PersistentClient(path=self.db_path, settings=settings)
        self.collection = self.client.get_or_create_collection(
            name="image_vectors",
            metadata={
                "description": "CLIP image embeddings",
                "hnsw:space": "cosine"
            }
        )
    
    def encode_image(self, image_path: str) -> np.ndarray:
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def encode_text(self, text: str) -> np.ndarray:
        try:
            text_tokens = clip.tokenize([text]).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error encoding text '{text}': {e}")
            return None
    
    def add_image(self, image_path: str) -> bool:
        if not os.path.exists(image_path):
            return False
        
        try:
            stat = os.stat(image_path)
            file_size = stat.st_size
            modified_time = stat.st_mtime
            
            existing = self.collection.get(ids=[image_path])
            if existing['ids']:
                existing_metadata = existing['metadatas'][0]
                if existing_metadata.get('modified_time') == modified_time:
                    return True
            
            vector = self.encode_image(image_path)
            if vector is None:
                return False
            
            metadata = {
                "file_size": file_size,
                "modified_time": modified_time,
                "file_name": os.path.basename(image_path)
            }
            
            self.collection.upsert(
                ids=[image_path],
                embeddings=[vector.tolist()],
                metadatas=[metadata]
            )
            
            return True
        except Exception as e:
            print(f"Error adding image {image_path}: {e}")
            return False
    
    def build_database(self, folder_path: str, image_extensions: List[str] = None, progress_callback=None):
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic']
        
        image_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        print(f"Found {len(image_files)} image files")
        total_files = len(image_files)
        
        for i, image_path in enumerate(image_files):
            self.add_image(image_path)
            
            if progress_callback:
                progress = (i + 1) / total_files
                progress_callback(progress, i + 1, total_files, os.path.basename(image_path))
    
    def search_similar(self, query_image_path: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query_vector = self.encode_image(query_image_path)
        if query_vector is None:
            return []
        
        try:
            results = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=min(top_k + 1, self.collection.count()),
                include=['distances']
            )
            
            similarities = []
            for i, (file_path, cosine_distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
                if file_path == query_image_path:
                    continue
                
                cosine_similarity = 1 - cosine_distance
                similarities.append((file_path, float(cosine_similarity)))
            
            return similarities[:top_k]
        except Exception as e:
            print(f"Error searching similar images: {e}")
            return []
    
    def search_by_text(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query_vector = self.encode_text(query_text)
        if query_vector is None:
            return []
        
        try:
            results = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=min(top_k, self.collection.count()),
                include=['distances']
            )
            
            similarities = []
            for i, (file_path, cosine_distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
                cosine_similarity = 1 - cosine_distance
                similarities.append((file_path, float(cosine_similarity)))
            
            return similarities[:top_k]
        except Exception as e:
            print(f"Error searching by text '{query_text}': {e}")
            return []
    
    def get_database_stats(self) -> dict:
        try:
            count = self.collection.count()
            return {"total_images": count}
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {"total_images": 0}
    
    def clear_database(self):
        """データベースをクリアし、新しい設定で再初期化します。"""
        try:
            # 既存のコレクションを削除
            self.client.delete_collection("image_vectors")
            # 新しい設定でコレクションを再作成
            self.init_database()
            return True
        except Exception as e:
            print(f"Error clearing database: {e}")
            return False