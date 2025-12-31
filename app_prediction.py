import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os
import plotly.express as px

# --- 1. 基本設定 ---
st.set_page_config(page_title="AI配置分析システム Ver.1.0", layout="wide")

@st.cache_resource
def load_ai_model():
    if os.path.exists('model.pkl'):
        try:
            with open('model.pkl', 'rb') as f:
                return pickle.load(f)
        except: return None
    return None

model = load_ai_model()

# --- 2. ユーティリティ ---
def to_half_width(text):
    if pd.isna(text): return ""
    table = str.maketrans('０１２３４５６７８９．', '0123456789.')
    return re.sub(r'[^\d\.]', '', str(text).translate(table))

def clean_num(val):
    s = to_half_width(val)
    match = re.search(r'(\d+\.?\d*)', s)
    return float(match.group(1)) if match else 0.0

# --- 3. 核心：データ解析エンジン ---
def build_clean_data(file):
    try:
        # 生読み込み（名前の混乱を避けるためヘッダーなし）
        if file.name.endswith('.xlsx'):
            raw = pd.read_excel(file, header=None, engine='openpyxl')
        else:
            try: raw = pd.read_csv(file, header=None, encoding='utf-8')
            except: raw = pd.read_csv(file, header=None, encoding='cp932')
        
        # 見出し行を自動探索
        best_row, max_hits = 0, 0
        keywords = ['場所', 'R', '馬名', 'オッズ']
        for i in range(min(len(raw), 25)):
            row_vals = [str(x) for x in raw.iloc[i].values]
            hits = sum(1 for k in keywords if any(k in v for v in row_vals))
            if hits > max_hits:
                max_hits, best_row = hits, i
        
        headers = [str(x).strip() for x in raw.iloc[best_row].values]
        data_body = raw.iloc[best_row+1:].reset_index(drop=True)
        
        # --- 物理位置特定ロジック ---
        col_idx = {'正番': 5} # 【絶対指示】F列(6番目)は正番
        
        mapping = {
            '場名': ['場所', '場名', '会場'],
            'R': ['R', 'レース', '番組', 'Ｒ'],
            '馬名': ['馬名', '名称'],
            '単ｵｯｽﾞ': ['単勝', 'オッズ', '単ｵｯｽﾞ'],
            '騎手': ['騎手'], '厩舎': ['厩舎', '調教'], '馬主': ['馬主'],
            '着順': ['着順', '着', '結果']
        }
        
        for internal, keys in mapping.items():
            if internal in col_idx: continue
            for idx, h_name in enumerate(headers):
                if any(k in h_name for k in keys):
                    col_idx[internal] = idx; break

        # 新しいクリーンな表の構築
        df = pd.DataFrame(index=data_body.index)
        for col_name in ['R', '正番', '場名', '馬名', '単ｵｯｽﾞ', '騎手', '厩舎', '馬主', '着順']:
            idx = col_idx.get(col_name)
            if idx is not None and idx < len(data_body.columns):
                df[col_name] = data_body.iloc[:, idx]
            else:
                df[col_name] = np.nan if col_name == '着順' else (0 if col_name in ['R','正番'] else (99.0 if col_name == '単ｵｯｽﾞ' else "不明"))

        # データの整形
        df['R'] = df['R'].apply(clean_num).astype(int)
        df['正番'] = df['正番'].apply(clean_num).astype(int)
        df['単ｵｯｽﾞ'] = df['単ｵｯｽﾞ'].apply(clean_num).replace(0, 99.0)
        df = df[df['R'] > 0].copy()

        # --- 配置・ペア判定 ---
        df['青塗フラグ'] = 0; df['ペアフラグ'] = 0; df['判定理由'] = ""
        df['頭数'] = df.groupby(['場名', 'R'])['正番'].transform('max').fillna(16)
        df['逆番'] = (df['頭数'] + 1) - df['正番']
        df['正循環'] = df['頭数'] + df['正番']
        df['逆循環'] = df['頭数'] + df['逆番']

        for col in ['騎手', '厩舎', '馬主']:
            if col not in df.columns: continue
            for name, group in df.groupby(['場名', col] if col=='騎手' else col):
                if len(group) < 2 or str(name) in ['nan', '不明', '']: continue
                
                rows = group.sort_values('R').to_dict('records')
                indices = group.sort_values('R').index
                
                for i in range(len(rows)):
                    curr = rows[i]
                    curr_sets = {curr['正番'], curr['逆番'], curr['正循環'], curr['逆循環']}
                    
                    # 青塗判定
                    for j in range(len(rows)):
                        if i == j: continue
                        if curr_sets.intersection({rows[j]['正番'], rows[j]['逆番'], rows[j]['正循環'], rows[j]['逆循環']}):
                            df.at[indices[i], '青塗フラグ'] = 1
                            if f"★{col}青塗" not in df.at[indices[i], '判定理由']:
                                df.at[indices[i], '判定理由'] += f"★{col}青塗 "
                            break
                    
                    # ペア判定（連続レース）
                    if i > 0:
                        prev = rows[i-1]
                        if curr['R'] == prev['R'] + 1:
                            if curr_sets.intersection({prev['正番'], prev['逆番'], prev['正循環'], prev['逆循環']}):
                                for k in [indices[i], indices[i-1]]:
                                    df.at[k, 'ペアフラグ'] = 1
                                    if "★ペア" not in df.at[k, '判定理由']:
                                        df.at[k, '判定理由'] += "★ペア "

        # AI予測
        df['AI激走確率'] = 0.0
        if model:
            try:
                X = df[['単ｵｯｽﾞ', '青塗フラグ']].copy()
                X['ペアフラグ'] = df['ペアフラグ']
                X['前日フラグ'] = 0
                df['AI激走確率'] = [round(p[1] * 100, 1) for p in model.predict_proba(X)]
            except: pass

        return df, "success"
    except Exception as e:
        import traceback
        return pd.DataFrame(), traceback.format_exc()

# --- 4. UI ---
st.title("🏇 AI配置馬券分析システム：再構築版")

up_file = st.sidebar.file_uploader("配置表をアップロード", type=['xlsx', 'csv'])

if up_file:
    if 'df' not in st.session_state or st.sidebar.button("🔄 データを再解析"):
        res, status = build_clean_data(up_file)
        if status == "success": st.session_state['df'] = res
        else: st.error("解析失敗"); st.code(status)

    if 'df' in st.session_state:
        df = st.session_state['df'].copy()
        df['当日バイアス'] = 0.0
        
        # 着順入力による流動計算
        if '着順' in df.columns:
            df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
            hits = df[df['着順'] <= 3]
            if not hits.empty:
                hit_rate = len(hits[(hits['青塗フラグ']==1)|(hits['ペアフラグ']==1)]) / len(hits)
                df.loc[(df['青塗フラグ']==1)|(df['ペアフラグ']==1), '当日バイアス'] = hit_rate * 20.0

        df['期待値'] = df['AI激走確率'] + df['当日バイアス']
        
        places = sorted([p for p in df['場名'].unique() if str(p) not in ['nan','不明']])
        if places:
            target_p = st.sidebar.selectbox("会場", places)
            r_list = sorted(df[df['場名'] == target_p]['R'].unique())
            target_r = st.sidebar.selectbox("レース", r_list)

            view = df[(df['場名'] == target_p) & (df['R'] == target_r)].sort_values('期待値', ascending=False)
            
            st.subheader(f"📊 {target_p} {int(target_r)}R 激走予測")
            
            cols = ['正番', '馬名', '判定理由', 'AI激走確率', '期待値', '着順']
            final_cols = [c for c in cols if c in view.columns]
            
            ed = st.data_editor(view[final_cols], key=f"ed_{target_p}_{target_r}", 
                                hide_index=True, use_container_width=True,
                                column_config={
                                    "期待値": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
                                    "正番": st.column_config.NumberColumn(format="%d")
                                })
            
            if st.button("🔄 入力した着順を保存して更新"):
                for _, row in ed.iterrows():
                    st.session_state['df'].loc[(st.session_state['df']['場名']==target_p) & 
                                               (st.session_state['df']['R']==target_r) & 
                                               (st.session_state['df']['正番']==row['正番']), '着順'] = row['着順']
                st.rerun()
        else:
            st.error("会場（場所）が特定できませんでした。")
