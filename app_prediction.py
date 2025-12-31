import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os
import plotly.express as px

# --- 1. 基本設定 ---
st.set_page_config(page_title="AI配置分析システム", layout="wide")

@st.cache_resource
def load_ai_model():
    if os.path.exists('model.pkl'):
        try:
            with open('model.pkl', 'rb') as f:
                return pickle.load(f)
        except: return None
    return None

model = load_ai_model()

def to_half_width(text):
    if pd.isna(text): return ""
    table = str.maketrans('０１２３４５６７８９．', '0123456789.')
    return re.sub(r'[^\d\.]', '', str(text).translate(table))

def normalize_name(x):
    if pd.isna(x): return ''
    s = str(x).strip().replace('　', '').replace(' ', '')
    return re.split(r'[,(（/]', s)[0]

# --- 2. データ解析エンジン（ペア判定ロジック追加） ---
def load_and_analyze_with_pairs(file):
    try:
        if file.name.endswith('.xlsx'):
            df_raw = pd.read_excel(file, header=None, engine='openpyxl')
        else:
            try: df_raw = pd.read_csv(file, header=None, encoding='utf-8')
            except: df_raw = pd.read_csv(file, header=None, encoding='cp932')
        
        # ヘッダー行探索
        best_row, max_hits = 0, 0
        keywords = ['場所', 'R', '馬名', 'オッズ']
        for i in range(min(len(df_raw), 25)):
            row_vals = [str(x) for x in df_raw.iloc[i].values]
            hits = sum(1 for k in keywords if any(k in v for v in row_vals))
            if hits > max_hits:
                max_hits, best_row = hits, i
        
        header_names = [str(x).strip() for x in df_raw.iloc[best_row].values]
        df_data = df_raw.iloc[best_row+1:].reset_index(drop=True)
        
        col_idx = {'正番': 5 if len(df_data.columns) > 5 else None}
        mapping = {
            '場名': ['場所', '場名', '会場'], 'R': ['R', 'レース', '番組'],
            '馬名': ['馬名', '名称'], '単ｵｯｽﾞ': ['単勝', 'オッズ', '単ｵｯｽﾞ'],
            '騎手': ['騎手'], '厩舎': ['厩舎', '調教'], '馬主': ['馬主'], '着順': ['着順', '着']
        }
        for internal, keys in mapping.items():
            if internal in col_idx: continue
            for idx, name in enumerate(header_names):
                if any(k in str(name) for k in keys): col_idx[internal] = idx; break

        df = pd.DataFrame(index=df_data.index)
        for internal, idx in col_idx.items():
            if idx is not None: df[internal] = df_data.iloc[:, idx]
        
        # 数値クリーンアップ
        def clean_num(val):
            s = to_half_width(val)
            match = re.search(r'(\d+\.?\d*)', str(s))
            return float(match.group(1)) if match else 0.0

        for c in ['R', '正番', '単ｵｯｽﾞ']:
            if c in df.columns: df[c] = df[c].apply(clean_num)
        df['R'] = df['R'].astype(int)
        df['正番'] = df['正番'].astype(int)
        
        df = df[df['R'] > 0].copy()
        df['青塗フラグ'] = 0; df['ペアフラグ'] = 0; df['配置条件'] = ""
        
        # 基本配置計算
        df['頭数'] = df.groupby(['場名', 'R'])['正番'].transform('max')
        df['逆番'] = (df['頭数'] + 1) - df['正番']
        df['正循環'] = df['頭数'] + df['正番']
        df['逆循環'] = df['頭数'] + df['逆番']
        
        # --- ペア & 青塗 判定ロジック ---
        for col in ['騎手', '厩舎', '馬主']:
            if col not in df.columns: continue
            df[col] = df[col].astype(str).apply(normalize_name)
            
            # 属性ごとにソートしてペアを探す
            for attr_name, group in df.groupby(['場名', col] if col=='騎手' else col):
                if len(group) < 2 or attr_name in ['nan', '不明', '']: continue
                
                # 配置セットの作成
                sorted_group = group.sort_values('R')
                rows = sorted_group.to_dict('records')
                
                for i in range(len(rows)):
                    current = rows[i]
                    curr_sets = {current['正番'], current['逆番'], current['正循環'], current['逆循環']}
                    
                    # 青塗判定（同一条件内での一致）
                    is_blue = False
                    for j in range(len(rows)):
                        if i == j: continue
                        target_sets = {rows[j]['正番'], rows[j]['逆番'], rows[j]['正循環'], rows[j]['逆循環']}
                        if curr_sets.intersection(target_sets):
                            is_blue = True; break
                    
                    if is_blue:
                        idx = sorted_group.index[i]
                        df.at[idx, '青塗フラグ'] = 1
                        if f"★{col}青塗" not in df.at[idx, '配置条件']:
                            df.at[idx, '配置条件'] += f"★{col}青塗 "
                    
                    # ペア判定（隣接レース R と R+1 での一致）
                    if i > 0:
                        prev = rows[i-1]
                        if current['場名'] == prev['場名'] and current['R'] == prev['R'] + 1:
                            prev_sets = {prev['正番'], prev['逆番'], prev['正循環'], prev['逆循環']}
                            if curr_sets.intersection(prev_sets):
                                # 現在の行と前の行の両方にペアフラグを立てる
                                for offset in [0, -1]:
                                    idx = sorted_group.index[i + offset]
                                    df.at[idx, 'ペアフラグ'] = 1
                                    if "★ペア" not in df.at[idx, '配置条件']:
                                        df.at[idx, '配置条件'] += "★ペア "

        # AI予測（ペアフラグを考慮）
        df['AI激走確率'] = 0.0
        if model:
            try:
                # ペアがある場合はスコアを上乗せして渡す
                X = df[['単ｵｯｽﾞ', '青塗フラグ']].copy()
                X['ペアフラグ'] = df['ペアフラグ']
                X['前日フラグ'] = 0
                probs = model.predict_proba(X)
                df['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
            except: pass

        return df, "success"
    except Exception as e:
        return pd.DataFrame(), str(e)

# --- 3. UI ---
st.title("🏇 AI配置分析：ペア判定・流動統合版")

up_file = st.sidebar.file_uploader("配置表アップロード", type=['xlsx', 'csv'])

if up_file:
    if 'df' not in st.session_state or st.sidebar.button("🔄 再解析"):
        res, status = load_and_analyze_with_pairs(up_file)
        if status == "success": st.session_state['df'] = res
        else: st.error(f"解析失敗: {status}")

    if 'df' in st.session_state:
        df = st.session_state['df'].copy()
        
        # 当日バイアス計算
        df['当日バイアス'] = 0.0
        if '着順' in df.columns:
            df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
            hits = df[df['着順'] <= 3]
            if not hits.empty:
                # 青塗やペアが来ている割合でボーナス
                blue_hit = len(hits[hits['青塗フラグ'] == 1])
                pair_hit = len(hits[hits['ペアフラグ'] == 1])
                df.loc[df['青塗フラグ'] == 1, '当日バイアス'] += (blue_hit / len(hits)) * 10.0
                df.loc[df['ペアフラグ'] == 1, '当日バイアス'] += (pair_hit / len(hits)) * 10.0

        df['期待値'] = df['AI激走確率'] + df['当日バイアス']
        
        places = sorted([p for p in df['場名'].unique() if str(p) not in ['nan','不明']])
        if places:
            target_p = st.sidebar.selectbox("会場", places)
            r_list = sorted(df[df['場名'] == target_p]['R'].unique())
            target_r = st.sidebar.selectbox("レース", r_list)

            view = df[(df['場名'] == target_p) & (df['R'] == target_r)].sort_values('期待値', ascending=False)
            st.subheader(f"📊 {target_p} {int(target_r)}R 激走予測")
            
            # ペア条件が表示されるように列を構成
            disp_cols = ['正番', '馬名', '配置条件', 'AI激走確率', '期待値', '着順']
            final_cols = [c for c in disp_cols if c in view.columns]
            
            ed = st.data_editor(view[final_cols], key=f"ed_{target_p}_{target_r}", 
                                hide_index=True, use_container_width=True,
                                column_config={
                                    "期待値": st.column_config.ProgressColumn("最終期待値", format="%.1f", min_value=0, max_value=100),
                                    "正番": st.column_config.NumberColumn("馬番", format="%d")
                                })
            
            if st.button("🔄 着順を保存して更新"):
                for _, row in ed.iterrows():
                    st.session_state['df'].loc[(st.session_state['df']['場名']==target_p) & 
                                               (st.session_state['df']['R']==target_r) & 
                                               (st.session_state['df']['正番']==row['正番']), '着順'] = row['着順']
                st.rerun()
