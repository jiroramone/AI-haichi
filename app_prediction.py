import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os
import plotly.express as px

# --- 1. 基本設定 ---
st.set_page_config(page_title="AI配置分析：ペア・青塗分離版", layout="wide")

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

def clean_num(val):
    s = to_half_width(val)
    match = re.search(r'(\d+\.?\d*)', s)
    return float(match.group(1)) if match else 0.0

# --- 2. データ解析エンジン（ペアと青塗を分離） ---
def load_and_analyze_with_precision(file):
    try:
        if file.name.endswith('.xlsx'):
            raw = pd.read_excel(file, header=None, engine='openpyxl')
        else:
            try: raw = pd.read_csv(file, header=None, encoding='utf-8')
            except: raw = pd.read_csv(file, header=None, encoding='cp932')
        
        # ヘッダー探索
        best_row, max_hits = 0, 0
        keywords = ['場所', 'Ｒ', '馬名', '正番']
        for i in range(min(len(raw), 25)):
            row_vals = [str(x) for x in raw.iloc[i].values]
            hits = sum(1 for k in keywords if any(k in v for v in row_vals))
            if hits > max_hits:
                max_hits, best_row = hits, i
        
        headers = [str(x).strip() for x in raw.iloc[best_row].values]
        df_data = raw.iloc[best_row+1:].reset_index(drop=True)
        
        # 列特定（F列は絶対正番）
        col_idx = {'正番': 5}
        mapping = {
            '場名': ['場所', '場名'], 'R': ['Ｒ', 'R', 'レース'],
            '馬名': ['馬名'], '単ｵｯｽﾞ': ['単ｵｯｽﾞ', 'オッズ'],
            '騎手': ['騎手'], '厩舎': ['厩舎', '調教', '調教師'], '馬主': ['馬主'], '着順': ['着順']
        }
        for internal, keys in mapping.items():
            if internal in col_idx: continue
            for idx, h_name in enumerate(headers):
                if any(k in str(h_name) for k in keys):
                    col_idx[idx] = internal; break

        df = pd.DataFrame()
        for idx, name in col_idx.items():
            if idx < len(df_data.columns): df[name] = df_data.iloc[:, idx]
        
        df['R'] = df['R'].apply(clean_num).astype(int)
        df['正番'] = df['正番'].apply(clean_num).astype(int)
        df['単ｵｯｽﾞ'] = df['単ｵｯｽﾞ'].apply(clean_num).replace(0, 99.0)
        df = df[df['R'] > 0].copy()

        # --- 配置・ペア 厳密分離判定 ---
        df['青塗フラグ'] = 0; df['ペアフラグ'] = 0; df['判定理由'] = ""; df['属性'] = ""
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
                    
                    found_pair = False
                    found_blue = False
                    
                    for j in range(len(rows)):
                        if i == j: continue
                        target = rows[j]
                        target_sets = {target['正番'], target['逆番'], target['正循環'], target['逆循環']}
                        
                        if curr_sets.intersection(target_sets):
                            # レース間隔による判定の分離
                            if abs(curr['R'] - target['R']) == 1:
                                found_pair = True
                            else:
                                found_blue = True
                    
                    idx = indices[i]
                    df.at[idx, '属性'] += f"{col}:{name} / "
                    if found_pair:
                        df.at[idx, 'ペアフラグ'] = 1
                        df.at[idx, '判定理由'] += f"★{col}ペア "
                    elif found_blue: # ペアがない場合のみ青塗として表示
                        df.at[idx, '青塗フラグ'] = 1
                        df.at[idx, '判定理由'] += f"★{col}青塗 "

        # AI予測
        df['AI激走確率'] = 0.0
        if model:
            try:
                X = df[['単ｵｯｽﾞ', '青塗フラグ']].copy()
                X['ペアフラグ'] = df['ペアフラグ']; X['前日フラグ'] = 0
                df['AI激走確率'] = [round(p[1] * 100, 1) for p in model.predict_proba(X)]
            except: pass

        return df, "success"
    except Exception as e:
        import traceback
        return pd.DataFrame(), traceback.format_exc()

# --- 3. 流動的ペア連動ロジック ---
def apply_pair_link_logic(df):
    df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
    df['ペア連動ボーナス'] = 0.0
    
    #的中したペア馬の(場名, R, 属性)を抽出
    hit_pair_keys = set()
    hits = df[df['着順'] <= 3]
    for _, row in hits.iterrows():
        if "ペア" in str(row['判定理由']):
            for attr in str(row['属性']).split(' / '):
                if attr: hit_pair_keys.add((row['場名'], row['R'], attr))

    # 後続レースの相方にボーナスを反映
    for idx, row in df.iterrows():
        if "ペア" in str(row['判定理由']):
            for attr in str(row['属性']).split(' / '):
                if (row['場名'], row['R'] - 1, attr) in hit_pair_keys:
                    df.at[idx, 'ペア連動ボーナス'] = 20.0
                    break
    
    df['最終期待値'] = df['AI激走確率'] + df['ペア連動ボーナス']
    return df

# --- 4. UI ---
st.title("🏇 AI配置分析：ペア・青塗完全分離版")

up_file = st.sidebar.file_uploader("配置表アップロード", type=['xlsx', 'csv'])

if up_file:
    if 'main_df' not in st.session_state or st.sidebar.button("🔄 再解析"):
        res, status = load_and_analyze_with_precision(up_file)
        if status == "success": st.session_state['main_df'] = res
        else: st.error("解析失敗"); st.code(status)

    if 'main_df' in st.session_state:
        df = apply_pair_link_logic(st.session_state['main_df'].copy())
        
        places = sorted([p for p in df['場名'].unique() if str(p) not in ['nan','不明']])
        if places:
            target_p = st.sidebar.selectbox("会場を選択", places)
            r_list = sorted(df[df['場名'] == target_p]['R'].unique())
            target_r = st.sidebar.selectbox("レースを選択", r_list)

            view = df[(df['場名'] == target_p) & (df['R'] == target_r)].sort_values('最終期待値', ascending=False)
            st.subheader(f"📊 {target_p} {int(target_r)}R 予測結果")
            
            top = view.iloc[0]
            if top['ペア連動ボーナス'] > 0:
                st.warning(f"🚀 【ペア連動】 {int(top['正番'])}番 {top['馬名']} が連動により期待値上昇中！")

            cols = ['正番', '馬名', '判定理由', 'AI激走確率', '最終期待値', '着順']
            final_cols = [c for c in cols if c in view.columns]
            
            ed = st.data_editor(view[final_cols], key=f"ed_{target_p}_{target_r}", 
                                hide_index=True, use_container_width=True,
                                column_config={
                                    "最終期待値": st.column_config.ProgressColumn("最終期待値", format="%.1f", min_value=0, max_value=100),
                                    "正番": st.column_config.NumberColumn("馬番", format="%d")
                                })
            
            if st.button("🔄 着順を保存して次レースのペア期待値を更新"):
                for _, row in ed.iterrows():
                    st.session_state['main_df'].loc[(st.session_state['main_df']['場名']==target_p) & 
                                               (st.session_state['main_df']['R']==target_r) & 
                                               (st.session_state['main_df']['正番']==row['正番']), '着順'] = row['着順']
                st.rerun()
