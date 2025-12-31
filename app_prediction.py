import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os

# --- 1. 基本設定 ---
st.set_page_config(page_title="配置予測・最終安定版", layout="wide")

# モデル読込
@st.cache_resource
def get_model():
    MODEL_PATH = 'model.pkl'
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except: return None
    return None

model = get_model()

# --- 2. 文字・名前の整理 ---
def clean_val(x):
    if pd.isna(x): return ""
    return str(x).strip().replace('　', '').replace(' ', '')

def to_num(x):
    if pd.isna(x): return 0
    val = re.sub(r'[^\d\.]', '', str(x))
    return float(val) if '.' in val else int(float(val)) if val else 0

# --- 3. データ読み込み（失敗しない設計） ---
def safe_load(file):
    try:
        if file.name.endswith('.xlsx'): df = pd.read_excel(file)
        else: df = pd.read_csv(file, encoding='cp932') if 'csv' in file.name else pd.read_csv(file)
        
        # 項目名を探す
        for i in range(min(len(df), 20)):
            row = [str(x) for x in df.iloc[i].values]
            if any(k in "".join(row) for k in ['場所', 'R', '番', '馬']):
                df.columns = df.iloc[i]
                df = df.iloc[i+1:].reset_index(drop=True)
                break
        
        df.columns = [str(c).strip() for c in df.columns]
        
        # 名寄せ
        name_map = {'場所':'場名','開催':'場名','競馬場':'場名','レース':'R','番':'正番','馬番':'正番','オッズ':'単ｵｯｽﾞ','着':'着順'}
        for old, new in name_map.items():
            for c in df.columns:
                if old in str(c): df = df.rename(columns={c: new})

        # 必須列がなければ空で作る
        for c in ['場名','R','正番','馬名','単ｵｯｽﾞ']:
            if c not in df.columns: df[c] = "" if c in ['場名','馬名'] else 0

        # 型の整理
        df['R'] = df['R'].apply(to_num).astype(int)
        df['正番'] = df['正番'].apply(to_num).astype(int)
        df['単ｵｯｽﾞ'] = df['単ｵｯｽﾞ'].apply(to_num).astype(float)
        
        return df[df['R'] > 0].copy() # 1R以上のデータのみ
    except Exception as e:
        st.error(f"読込エラー: {e}")
        return pd.DataFrame()

# --- 4. 解析・AI予測（絶対止まらない） ---
def run_analysis(df):
    # 初期フラグ
    df['青塗フラグ'] = 0; df['ペアフラグ'] = 0; df['前日配置フラグ'] = 0; df['AI激走確率'] = 0.0
    
    # 簡単な青塗判定（重複チェック）
    for col in ['騎手', '厩舎', '馬主']:
        if col in df.columns:
            # 会場・Rをまたいで同じ名前がいればフラグ（超簡易版）
            dup = df[df[col] != ""].duplicated(subset=['場名', col], keep=False)
            df.loc[df[col] != "", '青塗フラグ'] = dup.astype(int)

    # AI予測（特徴量を厳格に合わせる）
    if model:
        try:
            # あなたが学習させた項目名に合わせる必要があります
            # ここでは一般的な4項目を想定
            X = pd.DataFrame({
                '単ｵｯｽﾞ': df['単ｵｯｽﾞ'].fillna(99),
                '青塗フラグ': df['青塗フラグ'].fillna(0),
                'ペアフラグ': df['ペアフラグ'].fillna(0),
                '前日配置フラグ': df['前日配置フラグ'].fillna(0)
            })
            probs = model.predict_proba(X)
            df['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
        except Exception as e:
            st.sidebar.warning(f"AI予測スキップ: {e}")
            
    return df

# --- 5. UI ---
st.title("🏇 配置予測システム（最終安定版）")

file = st.sidebar.file_uploader("ファイルをアップロード")

if file:
    data = safe_load(file)
    if not data.empty:
        # 解析
        if 'final_df' not in st.session_state:
            st.session_state['final_df'] = run_analysis(data)
        
        final_df = st.session_state['final_df']
        
        # 選択UI
        places = sorted(final_df['場名'].unique())
        if places:
            pl = st.sidebar.selectbox("会場", places)
            rs = sorted(final_df[final_df['場名'] == pl]['R'].unique())
            r = st.sidebar.selectbox("レース", rs)
            
            # 表示
            view = final_df[(final_df['場名'] == pl) & (final_df['R'] == r)].sort_values('正番')
            
            st.subheader(f"📊 {pl} {r}R 結果表示 (全 {len(view)} 頭)")
            st.dataframe(
                view[['正番', '馬名', '単ｵｯｽﾞ', 'AI激走確率', '青塗フラグ', '着順']],
                column_config={"AI激走確率": st.column_config.ProgressColumn(min_value=0, max_value=100)},
                use_container_width=True, hide_index=True
            )
            
            st.divider()
            with st.expander("🔍 読み込みデータの詳細（デバッグ用）"):
                st.write("見つかった列名:", list(final_df.columns))
                st.write("データ件数:", len(final_df))
                st.dataframe(final_df.head())
        else:
            st.warning("会場名が正しく認識されていません。列名を確認してください。")
            st.write("認識された列:", list(data.columns))

if st.sidebar.button("🗑️ キャッシュを消去して再試行"):
    st.session_state.clear()
    st.rerun()
