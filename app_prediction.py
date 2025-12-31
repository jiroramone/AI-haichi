import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os
import plotly.express as px

# --- 1. 基本設定 ---
st.set_page_config(page_title="AI配置馬券 完全全頭予測", layout="wide")

MODEL_PATH = 'model.pkl'
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

def to_half_width(text):
    if pd.isna(text): return text
    table = str.maketrans('０１２３４５６７８９．', '0123456789.')
    return re.sub(r'[^\d\.]', '', str(text).translate(table))

def normalize_name(x):
    if pd.isna(x): return ''
    s = str(x).strip().replace('　', '').replace(' ', '')
    return re.split(r'[,(（/]', s)[0]

# --- 2. データ読み込み ---
def load_data(file):
    try:
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        else:
            try: df = pd.read_csv(file, encoding='utf-8')
            except: df = pd.read_csv(file, encoding='cp932')
        
        # ヘッダー探索
        for i in range(min(len(df), 20)):
            row_vals = [str(x) for x in df.iloc[i].values]
            if any(re.search(r'^[RrＲｒ]$|^場所$|^馬名$', str(x).strip()) for x in row_vals):
                df.columns = df.iloc[i]; df = df.iloc[i+1:].reset_index(drop=True); break

        df.columns = [str(c).strip() for c in df.columns]
        name_map = {'場所':'場名','開催':'場名','競馬場':'場名','レース':'R','Ｒ':'R','番':'正番','馬番':'正番','単勝オッズ':'単ｵｯｽﾞ','オッズ':'単ｵｯｽﾞ','単ｵｯズ':'単ｵｯｽﾞ'}
        df = df.rename(columns=name_map)
        
        df['R'] = pd.to_numeric(df['R'].apply(to_half_width), errors='coerce')
        df['正番'] = pd.to_numeric(df['正番'].apply(to_half_width), errors='coerce')
        df = df.dropna(subset=['R', '正番'])
        df['R'] = df['R'].astype(int); df['正番'] = df['正番'].astype(int)
        df['単ｵｯｽﾞ'] = pd.to_numeric(df['単ｵｯｽﾞ'].apply(to_half_width), errors='coerce').fillna(99.0)
        
        for col in ['騎手', '厩舎', '馬主', '馬名', '場名']:
            if col in df.columns: df[col] = df[col].astype(str).apply(normalize_name)
            else: df[col] = ""
            
        return df.copy()
    except: return pd.DataFrame()

# --- 3. 配置解析・全頭AI予測 ---
def run_full_analysis(df, df_prev=None):
    # 【最重要】全頭のフラグをまず「0」で初期化
    df['青塗フラグ'] = 0
    df['ペアフラグ'] = 0
    df['前日配置フラグ'] = 0
    df['スコア'] = 0.0
    df['タイプ'] = ""

    # 頭数・逆番計算
    df['頭数'] = df.groupby(['場名', 'R'])['正番'].transform('max').fillna(16).astype(int)
    df['逆番'] = (df['頭数'] + 1) - df['正番']
    df['正循環'] = df['頭数'] + df['正番']
    df['逆循環'] = df['頭数'] + df['逆番']

    idx_map = {(row['場名'], row['R'], row['正番']): idx for idx, row in df.iterrows()}

    # 青塗
    for col in ['騎手', '厩舎', '馬主']:
        g_keys = ['場名', col] if col == '騎手' else [col]
        for name, group in df.groupby(g_keys):
            if len(group) < 2 or not name: continue
            all_sets = [{r['正番'], r['逆番'], r['正循環'], r['逆循環']} for _, r in group.iterrows()]
            common = set.intersection(*all_sets)
            if common:
                for _, row in group.iterrows():
                    idx = idx_map.get((row['場名'], row['R'], row['正番']))
                    if idx is not None:
                        df.at[idx, '青塗フラグ'] = 1
                        df.at[idx, 'スコア'] += 9.0
                        df.at[idx, 'タイプ'] += f"★{col}青塗 "

    # ペア
    pair_labels = list("ABCDEFGHIJKLMNOP")
    for col in ['騎手', '厩舎', '馬主']:
        for name, group in df.groupby(['場名', col] if col=='騎手' else col):
            if len(group) < 2 or not name: continue
            rows = group.sort_values('R').to_dict('records')
            for i in range(len(rows)-1):
                r1, r2 = rows[i], rows[i+1]
                v1, v2 = [r1[c] for c in ['正番','逆番','正循環','逆循環']], [r2[c] for c in ['正番','逆番','正循環','逆循環']]
                if any(x in v2 for x in v1 if x != 0):
                    df.at[idx_map[(r1['場名'],r1['R'],r1['正番'])], 'ペアフラグ'] = 1
                    df.at[idx_map[(r2['場名'],r2['R'],r2['正番'])], 'ペアフラグ'] = 1
                    df.at[idx_map[(r1['場名'],r1['R'],r1['正番'])], 'タイプ'] += "◎ペア "
                    df.at[idx_map[(r2['場名'],r2['R'],r2['正番'])], 'タイプ'] += "◎ペア "

    # AI予測：全頭に対して一気に実行
    if model:
        # 学習時と同じ4つの特徴量を、全行分用意
        X = df[['単ｵｯｽﾞ', '青塗フラグ', 'ペアフラグ', '前日配置フラグ']].copy()
        # 欠損値があれば0で埋める
        X = X.fillna(0)
        
        # 確率算出
        probs = model.predict_proba(X)
        # 全行に確率を流し込む
        df['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
    else:
        df['AI激走確率'] = 0.0

    return df

# --- 4. UI 表示 ---
st.title("🏇 完全全頭表示 AI配置予測")

up_file = st.sidebar.file_uploader("当日配置表をアップロード", type=['xlsx', 'csv'])

if up_file:
    df_raw = load_data(up_file)
    if not df_raw.empty:
        # セッションに保存して状態を維持
        if 'full_df' not in st.session_state:
            st.session_state['full_df'] = run_full_analysis(df_raw)
        
        df = st.session_state['full_df']
        
        # 会場・レース選択
        places = sorted(df['場名'].unique())
        place = st.sidebar.selectbox("会場選択", places)
        r_nums = sorted(df[df['場名'] == place]['R'].unique())
        r_num = st.sidebar.selectbox("レース選択", r_nums)

        # 表示用データの抽出
        res = df[(df['場名'] == place) & (df['R'] == r_num)].sort_values('AI激走確率', ascending=False)
        
        st.subheader(f"📊 {place} {r_num}R 予測結果（全 {len(res)} 頭）")
        
        # 表の表示
        st.dataframe(
            res[['正番', '馬名', '単ｵｯｽﾞ', 'AI激走確率', 'タイプ', '着順']],
            column_config={
                "AI激走確率": st.column_config.ProgressColumn("AI激走確率", format="%.1f%%", min_value=0, max_value=100),
                "単ｵｯｽﾞ": st.column_config.NumberColumn("オッズ", format="%.1f")
            },
            use_container_width=True,
            hide_index=True
        )
        
        if st.button("🗑️ データをリセット"):
            del st.session_state['full_df']
            st.rerun()
