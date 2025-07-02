import streamlit as st
import os
from pathlib import Path
from PIL import Image
import tempfile
from clip_vector_db import CLIPVectorDB
from path_utils import normalize_path, validate_path_exists
import pillow_heif

# HEICをPILで開けるように登録
pillow_heif.register_heif_opener()

st.set_page_config(
    page_title="画像類似検索",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_db():
    return CLIPVectorDB()

def main():
    st.title("🔍 画像類似検索アプリ")
    st.markdown("CLIP ベクトルを使用した画像の類似検索システム")
    
    db = load_db()
    
    tabs = st.tabs(["🔍 類似検索", "📝 文字列で検索", "📁 データベース更新", "📊 統計情報"])
    
    with tabs[0]:
        st.header("画像類似検索")
        
        uploaded_file = st.file_uploader(
            "検索したい画像をアップロードしてください",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'heic']
        )
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            top_k = st.slider("表示する類似画像数", min_value=1, max_value=100, value=10)
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                query_image_path = tmp_file.name
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("検索画像")
                image = Image.open(query_image_path)
                st.image(image, caption="アップロードされた画像", use_container_width=True)
            
            with col2:
                st.subheader("類似画像検索中...")
                with st.spinner("CLIP ベクトルを計算しています..."):
                    similar_images = db.search_similar(query_image_path, top_k)
                
                if similar_images:
                    st.success(f"{len(similar_images)}件の類似画像が見つかりました")
                    
                    for i, (image_path, similarity) in enumerate(similar_images):
                        if os.path.exists(image_path):
                            try:
                                st.markdown(f"**{i+1}. 類似度: {similarity:.4f}**")
                                st.markdown(f"ファイル: `{image_path}`")
                                
                                similar_image = Image.open(image_path)
                                st.image(similar_image, caption=f"類似度: {similarity:.4f}", width=300)
                                st.markdown("---")
                            except Exception as e:
                                st.error(f"画像を読み込めませんでした: {image_path}")
                        else:
                            st.warning(f"ファイルが見つかりません: {image_path}")
                else:
                    st.warning("類似画像が見つかりませんでした。データベースを更新してください。")
            
            os.unlink(query_image_path)
    
    with tabs[1]:
        st.header("文字列で画像検索")
        
        query_text = st.text_input(
            "検索したい内容を入力してください",
            placeholder="例: 猫, 犬, 車, 花, 海, 山, 人物 など",
            help="日本語または英語で画像の内容を記述してください"
        )
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            top_k = st.slider("表示する画像数", min_value=1, max_value=100, value=10, key="text_search_top_k")
        
        if query_text:
            st.subheader(f"「{query_text}」の検索結果")
            
            with st.spinner("テキストをベクトル化して画像を検索しています..."):
                similar_images = db.search_by_text(query_text, top_k)
            
            if similar_images:
                st.success(f"{len(similar_images)}件の関連画像が見つかりました")
                
                # グリッド表示で画像を配置
                cols = st.columns(3)
                for i, (image_path, similarity) in enumerate(similar_images):
                    col_idx = i % 3
                    
                    with cols[col_idx]:
                        if os.path.exists(image_path):
                            try:
                                image = Image.open(image_path)
                                st.image(image, caption=f"類似度: {similarity:.4f}", use_container_width=True)
                                st.markdown(f"**ファイル:** `{os.path.basename(image_path)}`")
                                st.markdown(f"**パス:** `{image_path}`")
                                st.markdown("---")
                            except Exception as e:
                                st.error(f"画像を読み込めませんでした: {os.path.basename(image_path)}")
                        else:
                            st.warning(f"ファイルが見つかりません: {os.path.basename(image_path)}")
            else:
                st.warning("関連画像が見つかりませんでした。データベースを更新するか、別のキーワードを試してください。")
                st.info("💡 検索のコツ:\n- 簡潔な単語を使用（「猫」「車」「花」など）\n- 英語でも検索可能（「cat」「car」「flower」など）\n- 色や形容詞も有効（「赤い花」「大きな犬」など）")
    
    with tabs[2]:
        st.header("データベース更新")
        
        col1, col2 = st.columns(2)
        
        with col1:
            folder_path_input = st.text_input(
                "画像フォルダのパス",
                placeholder="C:\\Users\\user\\Pictures または /mnt/c/Users/user/Pictures",
                help="画像が保存されているフォルダのパスを入力してください（Windows形式のパスも自動変換されます）"
            )
            
            if folder_path_input:
                # Windows形式のパスをWSL形式に変換
                folder_path = normalize_path(folder_path_input)
                
                # 変換されたパスを表示
                if folder_path != folder_path_input:
                    st.info(f"パスを変換しました: `{folder_path}`")
                
                if validate_path_exists(folder_path):
                    st.success("✅ フォルダが見つかりました")
                    
                    if st.button("データベースを更新", type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        current_file = st.empty()
                        
                        def update_progress(progress, current, total, filename):
                            progress_bar.progress(progress)
                            status_text.text(f"処理中: {current}/{total} 枚")
                            current_file.text(f"現在の画像: {filename}")
                        
                        try:
                            db.build_database(folder_path, progress_callback=update_progress)
                            status_text.success("データベースの更新が完了しました！")
                            current_file.empty()
                        except Exception as e:
                            st.error(f"エラーが発生しました: {e}")
                            status_text.empty()
                            current_file.empty()
                else:
                    st.error("❌ フォルダが見つかりません")
                    st.info("💡 例: `C:\\Users\\user\\Pictures` または `/mnt/c/Users/user/Pictures`")
        
        with col2:
            st.subheader("操作")
            
            if st.button("データベースをクリア", type="secondary"):
                if st.checkbox("本当にデータベースをクリアしますか？"):
                    db.clear_database()
                    st.success("データベースがクリアされました")
                    st.rerun()
    
    with tabs[3]:
        st.header("データベース統計")
        
        stats = db.get_database_stats()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("登録画像数", stats["total_images"])
        
        with col2:
            if os.path.exists(db.db_path):
                db_size = sum(f.stat().st_size for f in Path(db.db_path).rglob('*') if f.is_file()) / (1024 * 1024)
                st.metric("DB サイズ", f"{db_size:.2f} MB")
        
        with col3:
            st.metric("使用モデル", "CLIP ViT-B/32")

if __name__ == "__main__":
    main()