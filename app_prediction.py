import streamlit as st
import pandas as pd
import os

# 1. 画面の基本設定
st.set_page_config(page_title="診断モード", layout="wide")
st.title("🔍 アプリ動作診断システム")

# 2. ファイルアップローダー
st.sidebar.header("設定")
file = st.sidebar.file_uploader("ファイルをアップロードしてください", type=['xlsx', 'csv'])

# 3. model.pklの存在確認（ここが原因で止まっていないかチェック）
if os.path.exists('model.pkl'):
    st.sidebar.success("✅ model.pkl を検出しました")
else:
    st.sidebar.warning("⚠️ model.pkl が見つかりません（GitHubにありますか？）")

# 4. メイン処理
if file:
    st.info(f"ファイル名: {file.name} を受け取りました。読み込みを開始します...")
    
    try:
        # 読み込み
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        else:
            try:
                df = pd.read_csv(file, encoding='utf-8')
            except:
                df = pd.read_csv(file, encoding='cp932')
        
        st.success("✅ ファイルの読み込みに成功しました！")
        
        # データの詳細を表示
        st.subheader("📊 読み込まれたデータ（生の状態）")
        st.write(f"データの行数: {len(df)}")
        st.write(f"認識された列名一覧: {list(df.columns)}")
        
        # 最初の5行を表示
        st.dataframe(df.head(20))
        
        # 配置計算に必要なキーワードが含まれているかチェック
        st.subheader("🧐 項目チェック")
        keywords = ['場所', 'R', '番', '馬名']
        for k in keywords:
            found = any(k in str(col) for col in df.columns)
            if found:
                st.write(f"・{k} 列： ✅ 発見")
            else:
                st.write(f"・{k} 列： ❌ 見つかりません（これが原因で計算が止まります）")

    except Exception as e:
        st.error(f"❌ 読み込み中にエラーが発生しました: {e}")
        st.write("エラーの型:", type(e))

else:
    st.write("👈 左側のサイドバーから、予測したいファイルをアップロードしてください。")

# 5. キャッシュリセット
if st.sidebar.button("🗑️ キャッシュを完全にリセット"):
    st.cache_data.clear()
    st.rerun()
