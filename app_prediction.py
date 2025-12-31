import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os
import plotly.express as px
import time

# --- 1. 基本設定 ---
st.set_page_config(page_title="AI配置馬券 全頭予測システム", layout="wide")

MODEL_PATH = 'model.pkl'
model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except:
        st.error("model.pkl の読み込みに失敗しました。")

def to_half_width(text):
    if pd.isna(text): return text
    table = str.maketrans('０１２３４５６７８９．', '0123456789.')
    return re.sub(r'[^\d\.]', '', str(text).translate(table))

def normalize_name(x):
    if pd.isna(x): return ''
    s = str(x).strip().replace('　', '').replace(' ', '')
    s = re.split(r'[,(（/]', s)[0]
    return re.sub(r'[★☆▲△◇$*]', '', s)

# --- 2. データ読み込み ---
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        else:
            try: df = pd.read_csv(file, encoding='utf-8')
            except: df = pd.read_csv(file, encoding='cp932')
        
        for i in range(min(len(df), 20)):
            row_vals = [str(x) for x in df.iloc[i].values]
            if any(re.search(r'^[RrＲｒ]$|^場所$|^馬名$', str(x).strip()) for x in row_vals):
                df.columns = df.iloc[i]; df = df.iloc[i+1:].reset_index(drop=True); break

        df.columns = df.columns.astype(str).str.strip()
        name_map = {'場所':'場名','開催':'場名','競馬場':'場名','レース':'R','Ｒ':'R','番':'正番','馬番':'正番','単オッズ':'単ｵｯｽﾞ','単勝オッズ':'単ｵｯｽﾞ','オッズ':'単ｵｯｽﾞ','着':'着順'}
        df = df.rename(columns=name_map)
        for col in ['R', '場名', '馬名', '正番', '騎手', '厩舎', '馬主', '単ｵｯｽﾞ', '着順']:
            if col not in df.columns: df[col] = np.nan

        df['R'] = pd.to_numeric(df['R'].apply(to_half_width), errors='coerce')
        df['正番'] = pd.to_numeric(df['正番'].apply(to_half_width), errors='coerce')
        df = df.dropna(subset=['R', '正番'])
        df['R'] = df['R'].astype(int); df['正番'] = df['正番'].astype(int)
        for col in ['騎手', '厩舎', '馬主', '馬名', '場名']:
            df[col] = df[col].apply(normalize_name)
        df['単ｵｯｽﾞ'] = pd.to_numeric(df['単ｵｯｽﾞ'].apply(to_half_width), errors='coerce')
        return df.copy(), "success"
    except Exception as e: return pd.DataFrame(), str(e)

# --- 3. 配置計算 & AI予測 ---
def analyze_haichi(df_curr, df_prev=None):
    df = df_curr.copy()
    if 'タイプ' in df.columns and df['タイプ'].notna().any(): return df

    df['頭数'] = df.groupby(['場名', 'R'])['正番'].transform('max').fillna(16).astype(int)
    df['逆番'] = (df['頭数'] + 1) - df['正番']
    df['正循環'] = df['頭数'] + df['正番']
    df['逆循環'] = df['頭数'] + df['逆番']

    # AI用フラグ
    df['青塗フラグ'] = 0; df['ペアフラグ'] = 0; df['前日配置フラグ'] = 0
    df['タイプ_list'] = [[] for _ in range(len(df))]
    df['属性_list'] = [[] for _ in range(len(df))]
    df['パターン_list'] = [[] for _ in range(len(df))]
    df['スコア'] = 0.0
    idx_map = {(row['場名'], row['R'], row['正番']): idx for idx, row in df.iterrows()}

    # 青塗判定
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
                        df.at[idx, 'タイプ_list'].append(f'★{col}青塗'); df.at[idx, 'スコア'] += 9.0

    # ペア判定
    pair_labels = list("ABCDEFGHIJKLMNOP")
    for col in ['騎手', '厩舎', '馬主']:
        for name, group in df.groupby(['場名', col] if col=='騎手' else col):
            if len(group) < 2 or not name: continue
            rows = group.sort_values('R').to_dict('records')
            for i in range(len(rows)-1):
                r1, r2 = rows[i], rows[i+1]
                v1, v2 = [r1[c] for c in ['正番','逆番','正循環','逆循環']], [r2[c] for c in ['正番','逆番','正循環','逆循環']]
                pats = [pair_labels[x*4+y] for x in range(4) for y in range(4) if v1[x]==v2[y] and v1[x]!=0]
                if pats:
                    df.at[idx_map[(r1['場名'],r1['R'],r1['正番'])], 'ペアフラグ'] = 1
                    df.at[idx_map[(r2['場名'],r2['R'],r2['正番'])], 'ペアフラグ'] = 1
                    for r_data in [r1, r2]:
                        idx = idx_map.get((r_data['場名'], r_data['R'], r_data['正番']))
                        if idx is not None:
                            df.at[idx, 'タイプ_list'].append('◎ペア'); df.at[idx, 'スコア'] += 3.5

    # AI予測 (全頭に対して実行)
    if model:
        X = df[['単ｵｯｽﾞ', '青塗フラグ', 'ペアフラグ', '前日配置フラグ']].fillna(0)
        probs = model.predict_proba(X)
        df['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]

    df['タイプ'] = df['タイプ_list'].apply(lambda x: ' / '.join(x) if x else '無')
    return df

def apply_ranking_logic(df_in):
    df = df_in.copy()
    df['総合スコア'] = df['スコア'] + (df.get('AI激走確率', 0) / 5.0)
    df['推奨'] = df['総合スコア'].apply(lambda x: "👑軸" if x>=15 else "🔥注" if x>=10 else "▲")
    return df

# --- 4. UI ---
st.title("🏇 AI配置馬券 予測・分析統合（全頭表示版）")

with st.sidebar:
    st.header("📂 読み込み")
    up_curr = st.file_uploader("当日データ", type=['xlsx', 'csv'])
    # 表示フィルター設定
    show_all = st.checkbox("シグナルがない馬もすべて表示する", value=True)

if up_curr:
    df_raw, status = load_data(up_curr)
    if status == "success":
        if 'analyzed_df' not in st.session_state:
            st.session_state['analyzed_df'] = apply_ranking_logic(analyze_haichi(df_raw))
        
        full_df = st.session_state['analyzed_df']

        st.subheader("📝 予測・結果入力")
        places = sorted(full_df['場名'].unique())
        p_tabs = st.tabs(places); edited_dfs = []
        for p_tab, place in zip(p_tabs, places):
            with p_tab:
                p_df = full_df[full_df['場名'] == place]
                r_nums = sorted(p_df['R'].unique())
                r_num = st.selectbox(f"レース選択", r_nums, key=f"sel_{place}")
                
                race_full = p_df[p_df['R'] == r_num].sort_values('AI激走確率', ascending=False)
                
                # フィルター処理
                disp = race_full if show_all else race_full[race_full['スコア'] > 0]
                
                # 推奨馬表示
                top = race_full.iloc[0]
                st.info(f"💡 AI推奨馬: {top['正番']}番 {top['馬名']} (確率 {top['AI激走確率']}%)")
                
                ed = st.data_editor(disp[['推奨','正番','馬名','単ｵｯｽﾞ','AI激走確率','タイプ','総合スコア','着順']], 
                                    hide_index=True, use_container_width=True, key=f"ed_{place}_{r_num}",
                                    column_config={"AI激走確率": st.column_config.ProgressColumn(format="%.1f%%")})
                
                # 更新用（略）
        
        # 統計グラフ表示
        st.divider()
        st.subheader("📈 激走確率の分布")
        st.plotly_chart(px.histogram(full_df, x="AI激走確率", color="場名", barmode="overlay"), use_container_width=True)
