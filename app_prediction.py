import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import pickle
import os

# --- 1. 基本設定 ---
st.set_page_config(page_title="配置馬券術 分析システム", layout="wide")

@st.cache_resource
def load_ai_model():
    MODEL_PATH = 'model.pkl'
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except: return None
    return None

model = load_ai_model()

def make_cols_unique(df):
    cols = []
    counts = {}
    for col in df.columns:
        c_str = str(col).strip() if pd.notna(col) else "Unnamed"
        if c_str in counts:
            counts[c_str] += 1
            cols.append(f"{c_str}_{counts[c_str]}")
        else:
            counts[c_str] = 0
            cols.append(c_str)
    df.columns = cols
    return df

def to_half_width(text):
    if pd.isna(text): return text
    table = str.maketrans('０１２３４５６７８９．', '0123456789.')
    return re.sub(r'[^\d\.]', '', str(text).translate(table))

# --- 2. データ読み込み（場所/F列対応） ---
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.xlsx'):
            df_raw = pd.read_excel(file, header=None, engine='openpyxl')
        else:
            try: df_raw = pd.read_csv(file, header=None, encoding='utf-8')
            except: df_raw = pd.read_csv(file, header=None, encoding='cp932')
        
        # 見出し行の自動探索
        best_row, max_hits = 0, 0
        keywords = ['場所', '場名', 'R', '馬名', '正番']
        for i in range(min(len(df_raw), 30)):
            row_str = "".join(df_raw.iloc[i].astype(str))
            hits = sum(1 for k in keywords if k in row_str)
            if hits > max_hits:
                max_hits, best_row = hits, i
        
        df = df_raw.iloc[best_row:].reset_index(drop=True)
        raw_headers = [str(c).strip() for c in df.iloc[0]]
        
        # 1. 強制設定: F列（6番目）を「正番」とする
        if len(raw_headers) >= 6:
            raw_headers[5] = "正番"
            
        df.columns = raw_headers
        df = df.iloc[1:].reset_index(drop=True)
        df = make_cols_unique(df)

        # 2. 列名の自動マッピング（「場所」を最優先で「場名」へ）
        mapping_rules = {
            '場名': ['場所', '場名', '競馬場', '開催'],
            'R': ['R', 'レース', '番組', 'Ｒ'],
            '馬名': ['馬名', '名称'],
            '単ｵｯｽﾞ': ['単ｵｯｽﾞ', '単勝オッズ', 'オッズ'],
            '着順': ['着順', '着', '結果', '順位'],
            '騎手': ['騎手', 'ジョッキー'],
            '厩舎': ['厩舎', '調教師'],
            '馬主': ['馬主', 'オーナー']
        }
        
        col_map = {}
        for internal, keys in mapping_rules.items():
            for c in df.columns:
                if c in col_map.keys(): continue
                # 完全一致または部分一致で判定
                if any(k == str(c) for k in keys) or any(k in str(c) for k in keys):
                    col_map[c] = internal
        
        df = df.rename(columns=col_map)
        df = make_cols_unique(df)

        # 3. 必須列の確保
        for col in ['R', '正番', '単ｵｯｽﾞ', '場名', '馬名', '着順']:
            if col not in df.columns:
                df[col] = np.nan if col == '着順' else (0 if col in ['R', '正番'] else (99.0 if col == '単ｵｯｽﾞ' else "不明"))

        # 4. データの型を整理（「1R」などの文字を除去）
        df['R'] = pd.to_numeric(df['R'].apply(to_half_width).astype(str).str.extract(r'(\d+)', expand=False), errors='coerce').fillna(0).astype(int)
        df['正番'] = pd.to_numeric(df['正番'].apply(to_half_width).astype(str).str.extract(r'(\d+)', expand=False), errors='coerce').fillna(0).astype(int)
        df['単ｵｯｽﾞ'] = pd.to_numeric(df['単ｵｯｽﾞ'].apply(to_half_width).astype(str).str.extract(r'(\d+\.?\d*)', expand=False), errors='coerce').fillna(99.0)
        
        # 会場名から余計な空白を消す
        df['場名'] = df['場名'].astype(str).str.strip().replace('nan', '不明')
        
        return df[df['R'] > 0].copy(), "success"
    except Exception as e: return pd.DataFrame(), str(e)

# --- 3. 解析エンジン ---
def analyze_haichi(df_curr):
    df = df_curr.copy()
    if 'タイプ' in df.columns and df['タイプ'].notna().any(): return df

    df['青塗フラグ'] = 0; df['ペアフラグ'] = 0; df['前日配置フラグ'] = 0; df['スコア'] = 0.0
    
    # 頭数・逆番計算
    df['頭数'] = df.groupby(['場名', 'R'])['正番'].transform('max').fillna(16).astype(int)
    df['逆番'] = (df['頭数'] + 1) - df['正番']
    df['正循環'] = df['頭数'] + df['正番']
    df['逆循環'] = df['頭数'] + df['逆番']
    
    idx_map = {(row['場名'], row['R'], row['正番']): idx for idx, row in df.iterrows()}
    
    # 青塗判定 (9.2点)
    for col in ['騎手', '厩舎', '馬主']:
        if col in df.columns:
            g_keys = ['場名', col] if col == '騎手' else [col]
            for name, group in df.groupby(g_keys):
                if len(group) < 2 or not str(name).strip() or str(name) == 'nan': continue
                all_sets = [{r['正番'], r['逆番'], r['正循環'], r['逆循環']} for _, r in group.iterrows()]
                common = set.intersection(*all_sets)
                if common:
                    for _, row in group.iterrows():
                        idx = idx_map.get((row['場名'], row['R'], row['正番']))
                        if idx is not None:
                            df.at[idx, '青塗フラグ'] = 1; df.at[idx, 'スコア'] += 9.2

    # AI予測
    if model:
        try:
            X = df[['単ｵｯｽﾞ', '青塗フラグ', 'ペアフラグ', '前日配置フラグ']].fillna(0)
            probs = model.predict_proba(X)
            df['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
        except: df['AI激走確率'] = 0.0
    else: df['AI激走確率'] = 0.0

    return df

def apply_ranking_logic(df_in):
    df = df_in.copy()
    df['総合スコア'] = df['スコア'] + (df.get('AI激走確率', 0) / 10.0)
    df['評価'] = df['総合スコア'].apply(lambda x: "👑軸" if x>=15 else "🔥注" if x>=10 else "▲")
    return df

# --- 4. UI ---
st.title("🏇 AI配置分析システム（場所・F列正番対応）")

with st.sidebar:
    st.header("📂 読み込み")
    up_curr = st.file_uploader("当日データ", type=['xlsx', 'csv'], key="curr")

if up_curr:
    df_raw, status = load_data(up_curr)
    if status == "success":
        if 'analyzed_df' not in st.session_state:
            st.session_state['analyzed_df'] = apply_ranking_logic(analyze_haichi(df_raw))
        
        full_df = st.session_state['analyzed_df']
        
        # 安全な会場リストの作成
        places = sorted([p for p in full_df['場名'].unique() if p and str(p) not in ['nan', '不明', 'None']])

        if places:
            st.subheader("📝 予測・結果入力")
            with st.form("result_form"):
                p_tabs = st.tabs(places)
                edited_dfs = []
                for p_tab, place in zip(p_tabs, places):
                    with p_tab:
                        p_df = full_df[full_df['場名'] == place]
                        r_nums = sorted(p_df['R'].unique())
                        if r_nums:
                            r_tabs = st.tabs([f"{r}R" for r in r_nums])
                            for r_tab, r_num in zip(r_tabs, r_nums):
                                with r_tab:
                                    race_full = p_df[p_df['R'] == r_num].sort_values('正番')
                                    target_cols = ['評価','正番','馬名','単ｵｯｽﾞ','AI激走確率','総合スコア','着順']
                                    disp_cols = [c for c in target_cols if c in race_full.columns]
                                    ed = st.data_editor(race_full[disp_cols], hide_index=True, use_container_width=True, key=f"ed_{place}_{r_num}",
                                                        column_config={"AI激走確率": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})
                                    updated = race_full.copy()
                                    for _, row in ed.iterrows():
                                        if '着順' in row: updated.loc[updated['正番'] == row['正番'], '着順'] = row['着順']
                                    edited_dfs.append(updated)
                
                if st.form_submit_button("🔄 入力を確定して更新"):
                    st.session_state['analyzed_df'] = apply_ranking_logic(pd.concat(edited_dfs, ignore_index=True))
                    st.rerun()
        else:
            st.error("会場（場所）を特定できませんでした。")
            st.write("現在認識されている列名:", list(full_df.columns))
            st.write("データプレビュー（最初の5行）:", full_df.head())
