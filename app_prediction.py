import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import pickle
import os

# --- 1. 基本設定 ---
st.set_page_config(page_title="配置分析システム・修正版", layout="wide")

# AIモデル読み込み
@st.cache_resource
def load_ai_model():
    MODEL_PATH = 'model.pkl'
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                m = pickle.load(f)
                return m
        except Exception as e:
            st.error(f"モデル読み込み失敗: {e}")
            return None
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

def normalize_name(x):
    if pd.isna(x): return ''
    s = str(x).strip().replace('　', '').replace(' ', '')
    return re.split(r'[,(（/]', s)[0]

# --- 2. データ読み込み（F列を馬番に、場所を場名に） ---
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.xlsx'):
            df_raw = pd.read_excel(file, header=None, engine='openpyxl')
        else:
            try: df_raw = pd.read_csv(file, header=None, encoding='utf-8')
            except: df_raw = pd.read_csv(file, header=None, encoding='cp932')
        
        # ヘッダー行を自動探索
        best_row, max_hits = 0, 0
        keywords = ['場所', 'R', '馬名']
        for i in range(min(len(df_raw), 30)):
            row_vals = [str(x) for x in df_raw.iloc[i].values]
            hits = sum(1 for k in keywords if k in "".join(row_vals))
            if hits > max_hits:
                max_hits, best_row = hits, i
        
        df = df_raw.iloc[best_row:].reset_index(drop=True)
        raw_headers = [str(c).strip() for c in df.iloc[0]]
        
        # 【重要】F列（インデックス5）を強制的に「正番」にする
        if len(raw_headers) >= 6:
            raw_headers[5] = "正番"
            
        df.columns = raw_headers
        df = df.iloc[1:].reset_index(drop=True)
        df = make_cols_unique(df)

        # 項目マッピング（「場所」を最優先で「場名」へ）
        mapping = {
            '場名': ['場所', '場名', '競馬場'],
            'R': ['R', 'レース', '番組'],
            '馬名': ['馬名', '名称'],
            '単ｵｯｽﾞ': ['単ｵｯｽﾞ', 'オッズ', '単勝'],
            '着順': ['着順', '結果', '順位'],
            '騎手': ['騎手', 'ジョッキー'],
            '厩舎': ['厩舎', '調教師'],
            '馬主': ['馬主', 'オーナー']
        }
        col_map = {}
        for internal, keys in mapping.items():
            for c in df.columns:
                if c in col_map.keys(): continue
                if any(k in str(c) for k in keys):
                    col_map[c] = internal; break
        
        df = df.rename(columns=col_map)
        
        # 数値クリーニング（正規表現をより確実に）
        def extract_num(val):
            s = to_half_width(val)
            match = re.search(r'(\d+\.?\d*)', str(s))
            return match.group(1) if match else "0"

        if 'R' in df.columns: df['R'] = df['R'].apply(extract_num).astype(float).astype(int)
        if '正番' in df.columns: df['正番'] = df['正番'].apply(extract_num).astype(float).astype(int)
        if '単ｵｯｽﾞ' in df.columns: df['単ｵｯｽﾞ'] = df['単ｵｯｽﾞ'].apply(extract_num).astype(float)
        
        for col in ['場名', '馬名', '騎手', '厩舎', '馬主']:
            if col in df.columns: df[col] = df[col].astype(str).apply(normalize_name)
            else: df[col] = "不明"

        return df[df['R'] > 0].copy(), "success"
    except Exception as e: return pd.DataFrame(), str(e)

# --- 3. 解析・AI予測（0%の原因を表示） ---
def analyze_haichi(df_curr):
    df = df_curr.copy()
    df['青塗フラグ'] = 0; df['ペアフラグ'] = 0; df['前日配置フラグ'] = 0; df['スコア'] = 0.0
    
    # 逆番などの配置計算
    df['頭数'] = df.groupby(['場名', 'R'])['正番'].transform('max').fillna(16).astype(int)
    df['逆番'] = (df['頭数'] + 1) - df['正番']
    df['正循環'] = df['頭数'] + df['正番']
    df['逆循環'] = df['頭数'] + df['逆番']
    
    idx_map = {(row['場名'], row['R'], row['正番']): idx for idx, row in df.iterrows()}
    
    # 青塗判定
    for col in ['騎手', '厩舎', '馬主']:
        if col in df.columns:
            g_keys = ['場名', col] if col == '騎手' else [col]
            for name, group in df.groupby(g_keys):
                if len(group) < 2 or str(name) in ['', '不明', 'nan']: continue
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
            # 学習時と同じ特徴量名・順番にする必要があります
            X = df[['単ｵｯｽﾞ', '青塗フラグ', 'ペアフラグ', '前日配置フラグ']].fillna(0)
            probs = model.predict_proba(X)
            df['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
        except Exception as e:
            # 0%になる原因を画面に表示する（デバッグ用）
            st.sidebar.error(f"AI予測エラー: {e}")
            df['AI激走確率'] = 0.0
    else:
        df['AI激走確率'] = 0.0

    return df

# --- 4. UI ---
st.title("🏇 AI配置分析システム（不具合修正版）")

up_curr = st.sidebar.file_uploader("当日データ", type=['xlsx', 'csv'], key="curr")

if up_curr:
    df_raw, status = load_data(up_curr)
    if status == "success":
        if 'analyzed_df' not in st.session_state:
            # 「着順」列がなくても動くようにapply_ranking_logicを統合
            analyzed = analyze_haichi(df_raw)
            analyzed['総合スコア'] = analyzed['スコア'] + (analyzed.get('AI激走確率', 0) / 10.0)
            analyzed['評価'] = analyzed['総合スコア'].apply(lambda x: "👑軸" if x>=15 else "🔥注" if x>=10 else "▲")
            st.session_state['analyzed_df'] = analyzed
        
        full_df = st.session_state['analyzed_df']
        # 会場名の特定を「場所」列からも対応
        places = sorted([p for p in full_df['場名'].unique() if p and str(p) not in ['不明', 'nan']])

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
                                    # 存在する列だけを表示
                                    target_cols = ['評価','正番','馬名','単ｵｯｽﾞ','AI激走確率','総合スコア','着順']
                                    disp_cols = [c for c in target_cols if c in race_full.columns]
                                    ed = st.data_editor(race_full[disp_cols], hide_index=True, use_container_width=True, key=f"ed_{place}_{r_num}",
                                                        column_config={"AI激走確率": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})
                                    edited_dfs.append(race_full) # 今回は更新ロジックを簡略化
                
                if st.form_submit_button("🔄 データを更新"):
                    st.rerun()
        else:
            st.error("会場（場所）が特定できませんでした。")
