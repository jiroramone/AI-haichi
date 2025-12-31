import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os

# --- 1. 基本設定 ---
st.set_page_config(page_title="AI配置予測・安定版", layout="wide")

@st.cache_resource
def load_ai_model():
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    return None

model = load_ai_model()

# --- 2. データ読み込み関数 ---
def load_raw_data(file):
    try:
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        else:
            try: df = pd.read_csv(file, encoding='utf-8')
            except: df = pd.read_csv(file, encoding='cp932')
        return df
    except Exception as e:
        st.error(f"ファイル読込エラー: {e}")
        return None

# --- 3. UIと処理 ---
st.title("🏇 配置予測システム（手動マッピング対応）")

up_file = st.sidebar.file_uploader("当日配置表(Excel/CSV)", type=['xlsx', 'csv'])

if up_file:
    df = load_raw_data(up_file)
    
    if df is not None:
        st.sidebar.success("✅ ファイル読込成功")
        
        # --- 列名の自動・手動マッピング ---
        st.sidebar.header("🔍 項目設定")
        all_cols = [str(c) for c in df.columns]
        
        # 自動検知ロジック
        def find_col(keywords, cols):
            for c in cols:
                if any(k in str(c) for k in keywords):
                    return c
            return cols[0]

        # ユーザーに確認・選択させる
        col_r = st.sidebar.selectbox("「R（レース番号）」はどの列ですか？", all_cols, index=all_cols.index(find_col(['R', 'Ｒ', 'レース', 'No'], all_cols)))
        col_place = st.sidebar.selectbox("「会場（場所）」はどの列ですか？", all_cols, index=all_cols.index(find_col(['場所', '場名', '会場', '開催'], all_cols)))
        col_num = st.sidebar.selectbox("「馬番（正番）」はどの列ですか？", all_cols, index=all_cols.index(find_col(['番', '馬番', '正番'], all_cols)))
        col_odds = st.sidebar.selectbox("「単勝オッズ」はどの列ですか？", all_cols, index=all_cols.index(find_col(['オッズ', '単勝', '単ｵｯｽﾞ'], all_cols)))

        if st.sidebar.button("🚀 この設定で解析実行"):
            # 内部的な名寄せ
            df = df.rename(columns={col_r: 'R', col_place: '場名', col_num: '正番', col_odds: '単ｵｯｽﾞ'})
            
            # データのクリーンアップ
            df['R'] = pd.to_numeric(df['R'].astype(str).str.extract(r'(\.0-9+)', expand=False), errors='coerce').fillna(0).astype(int)
            df['正番'] = pd.to_numeric(df['正番'].astype(str).str.extract(r'(\.0-9+)', expand=False), errors='coerce').fillna(0).astype(int)
            df['単ｵｯｽﾞ'] = pd.to_numeric(df['単ｵｯｽﾞ'].astype(str).str.extract(r'(\.0-9+)', expand=False), errors='coerce').fillna(99.0)
            
            # --- 解析処理 ---
            df['青塗フラグ'] = 0; df['ペアフラグ'] = 0; df['前日配置フラグ'] = 0; df['AI激走確率'] = 0.0
            
            # 簡易青塗判定
            for col in ['騎手', '厩舎', '馬主']:
                target_col = find_col([col], df.columns)
                if target_col in df.columns:
                    dup = df.duplicated(subset=['場名', target_col], keep=False)
                    df.loc[dup, '青塗フラグ'] = 1

            # AI予測
            if model:
                try:
                    features = ['単ｵｯｽﾞ', '青塗フラグ', 'ペアフラグ', '前日配置フラグ']
                    for f in features:
                        if f not in df.columns: df[f] = 0
                    X = df[features].fillna(0)
                    probs = model.predict_proba(X)
                    df['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
                except: pass

            st.session_state['result_df'] = df

    # --- 表示部 ---
    if 'result_df' in st.session_state:
        res_df = st.session_state['result_df']
        
        places = sorted(res_df['場名'].unique())
        place = st.selectbox("会場を選択", places)
        
        r_nums = sorted(res_df[res_df['場名'] == place]['R'].unique())
        r_num = st.selectbox("レースを選択", r_nums)
        
        view = res_df[(res_df['場名'] == place) & (res_df['R'] == r_num)].sort_values('AI激走確率', ascending=False)
        
        st.subheader(f"📊 {place} {r_num}R 予測結果")
        st.dataframe(
            view[['正番', 'AI激走確率', '単ｵｯｽﾞ', '青塗フラグ']],
            column_config={"AI激走確率": st.column_config.ProgressColumn(min_value=0, max_value=100)},
            use_container_width=True, hide_index=True
        )
else:
    st.info("👈 左側のサイドバーからファイルをアップロードしてください。")
