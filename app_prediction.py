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

# --- 2. データ解析エンジン（ペア・F列・R特定） ---
def load_and_analyze_data(file):
    try:
        # 生データ読み込み
        if file.name.endswith('.xlsx'):
            df_raw = pd.read_excel(file, header=None, engine='openpyxl')
        else:
            try: df_raw = pd.read_csv(file, header=None, encoding='utf-8')
            except: df_raw = pd.read_csv(file, header=None, encoding='cp932')
        
        # 見出し行探索
        best_row, max_hits = 0, 0
        keywords = ['場所', 'R', '馬名', 'オッズ']
        for i in range(min(len(df_raw), 30)):
            row_vals = [str(x) for x in df_raw.iloc[i].values]
            hits = sum(1 for k in keywords if any(k in v for v in row_vals))
            if hits > max_hits:
                max_hits, best_row = hits, i
        
        header_names = [str(x).strip() for x in df_raw.iloc[best_row].values]
        df_data = df_raw.iloc[best_row+1:].reset_index(drop=True)
        
        # --- 列の物理位置特定 ---
        col_idx = {'正番': 5 if len(df_data.columns) > 5 else None} # F列(6番目)固定
        
        mapping = {
            '場名': ['場所', '場名', '会場', '競馬場'],
            'R': ['R', 'レース', '番組', 'Ｒ'],
            '馬名': ['馬名', '名称'],
            '単ｵｯｽﾞ': ['単勝', 'オッズ', '単ｵｯｽﾞ'],
            '騎手': ['騎手'], '厩舎': ['厩舎', '調教'], '馬主': ['馬主'],
            '着順': ['着順', '着', '結果', '順位']
        }
        
        for internal, keys in mapping.items():
            if internal in col_idx: continue
            for idx, name in enumerate(header_names):
                if any(k in str(name) for k in keys):
                    col_idx[internal] = idx; break

        # クリーンなデータフレーム作成
        df = pd.DataFrame()
        # 必要な列をすべて確保（KeyError防止）
        for col in ['R', '正番', '場名', '馬名', '単ｵｯｽﾞ', '騎手', '厩舎', '馬主', '着順']:
            idx = col_idx.get(col)
            if idx is not None and idx < len(df_data.columns):
                df[col] = df_data.iloc[:, idx]
            else:
                df[col] = np.nan if col == '着順' else (0 if col in ['R','正番'] else (99.0 if col == '単ｵｯｽﾞ' else "不明"))

        # 数値クリーニング
        def clean_num(val):
            s = to_half_width(val)
            match = re.search(r'(\d+\.?\d*)', str(s))
            return float(match.group(1)) if match else 0.0

        df['R'] = df['R'].apply(clean_num).astype(int)
        df['正番'] = df['正番'].apply(clean_num).astype(int)
        df['単ｵｯｽﾞ'] = df['単ｵｯｽﾞ'].apply(clean_num).replace(0, 99.0)
        
        # --- 配置・ペア 判定ロジック ---
        df = df[df['R'] > 0].copy()
        df['青塗フラグ'] = 0; df['ペアフラグ'] = 0; df['判定'] = ""
        df['頭数'] = df.groupby(['場名', 'R'])['正番'].transform('max').fillna(16)
        df['逆番'] = (df['頭数'] + 1) - df['正番']
        df['正循環'] = df['頭数'] + df['正番']
        df['逆循環'] = df['頭数'] + df['逆番']
        
        for col in ['騎手', '厩舎', '馬主']:
            if col not in df.columns: continue
            df[col] = df[col].astype(str).apply(normalize_name)
            
            for attr, group in df.groupby(['場名', col] if col=='騎手' else col):
                if len(group) < 2 or attr in ['nan', '不明', '']: continue
                
                rows = group.sort_values('R').to_dict('records')
                group_indices = group.sort_values('R').index
                
                for i in range(len(rows)):
                    curr = rows[i]
                    curr_sets = {curr['正番'], curr['逆番'], curr['正循環'], curr['逆循環']}
                    
                    # 1. 青塗判定
                    for j in range(len(rows)):
                        if i == j: continue
                        if curr_sets.intersection({rows[j]['正番'], rows[j]['逆番'], rows[j]['正循環'], rows[j]['逆循環']}):
                            df.at[group_indices[i], '青塗フラグ'] = 1
                            if f"★{col}青塗" not in df.at[group_indices[i], '判定']:
                                df.at[group_indices[i], '判定'] += f"★{col}青塗 "
                            break
                    
                    # 2. ペア判定（隣接レース）
                    if i > 0:
                        prev = rows[i-1]
                        if curr['R'] == prev['R'] + 1:
                            if curr_sets.intersection({prev['正番'], prev['逆番'], prev['正循環'], prev['逆循環']}):
                                for idx in [group_indices[i], group_indices[i-1]]:
                                    df.at[idx, 'ペアフラグ'] = 1
                                    if "★ペア" not in df.at[idx, '判定']:
                                        df.at[idx, '判定'] += "★ペア "

        # AI基礎予測
        if model:
            try:
                X = df[['単ｵｯｽﾞ', '青塗フラグ']].copy()
                X['ペアフラグ'] = df['ペアフラグ']
                X['前日フラグ'] = 0
                probs = model.predict_proba(X)
                df['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
            except: df['AI激走確率'] = 0.0
        else: df['AI激走確率'] = 0.0

        return df, "success"
    except Exception as e:
        return pd.DataFrame(), str(e)

# --- 3. UI ---
st.title("🏇 AI配置分析：ペア判定・流動統合版")

up_file = st.sidebar.file_uploader("当日配置表(Excel/CSV)", type=['xlsx', 'csv'])

if up_file:
    if 'df' not in st.session_state or st.sidebar.button("🔄 データを再解析"):
        res, status = load_and_analyze_data(up_file)
        if status == "success": st.session_state['df'] = res
        else: st.error(f"解析失敗: {status}")

    if 'df' in st.session_state:
        df = st.session_state['df'].copy()
        df['当日バイアス'] = 0.0
        
        # 【流動的ロジック】結果による期待値補正
        if '着順' in df.columns:
            df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
            hits = df[df['着順'] <= 3]
            if not hits.empty:
                # 的中馬の青塗・ペア率でボーナス加算
                bias = (len(hits[(hits['青塗フラグ'] == 1) | (hits['ペアフラグ'] == 1)]) / len(hits)) * 20.0
                df.loc[(df['青塗フラグ'] == 1) | (df['ペアフラグ'] == 1), '当日バイアス'] = bias

        df['最終期待値'] = df['AI激走確率'] + df['当日バイアス']
        
        places = sorted([p for p in df['場名'].unique() if str(p) not in ['nan','不明']])
        if places:
            target_p = st.sidebar.selectbox("会場を選択", places)
            r_list = sorted(df[df['場名'] == target_p]['R'].unique())
            target_r = st.sidebar.selectbox("レースを選択", r_list)

            view = df[(df['場名'] == target_p) & (df['R'] == target_r)].sort_values('最終期待値', ascending=False)
            st.subheader(f"📊 {target_p} {int(target_r)}R 予測結果")
            
            disp_cols = ['正番', '馬名', '判定', 'AI激走確率', '当日バイアス', '最終期待値', '着順']
            final_cols = [c for c in disp_cols if c in view.columns]
            
            ed = st.data_editor(view[final_cols], key=f"ed_{target_p}_{target_r}", 
                                hide_index=True, use_container_width=True,
                                column_config={
                                    "最終期待値": st.column_config.ProgressColumn("期待値", format="%.1f", min_value=0, max_value=100),
                                    "正番": st.column_config.NumberColumn("馬番", format="%d")
                                })
            
            if st.button("🔄 着順を保存して今日の流れを反映"):
                for _, row in ed.iterrows():
                    st.session_state['df'].loc[(st.session_state['df']['場名']==target_p) & 
                                               (st.session_state['df']['R']==target_r) & 
                                               (st.session_state['df']['正番']==row['正番']), '着順'] = row['着順']
                st.rerun()
        else:
            st.error("会場を特定できません。")
else:
    st.info("左側のメニューからファイルを読み込んでください。")
