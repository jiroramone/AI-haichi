import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os
import plotly.express as px
import traceback

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

# --- 2. データ読み込み（正番の誤読防止ロジック） ---
def bulletproof_load(file):
    try:
        if file.name.endswith('.xlsx'):
            df_raw = pd.read_excel(file, header=None, engine='openpyxl')
        else:
            try: df_raw = pd.read_csv(file, header=None, encoding='utf-8')
            except: df_raw = pd.read_csv(file, header=None, encoding='cp932')
        
        # 見出し行を自動探索
        best_row, max_hits = 0, 0
        keywords = ['場所', 'R', '馬名', 'オッズ']
        for i in range(min(len(df_raw), 30)):
            row_vals = [str(x) for x in df_raw.iloc[i].values]
            hits = sum(1 for k in keywords if any(k in v for v in row_vals))
            if hits > max_hits:
                max_hits, best_row = hits, i
        
        headers = [str(x).strip() for x in df_raw.iloc[best_row].values]
        df_data = df_raw.iloc[best_row+1:].reset_index(drop=True)
        
        col_idx = {}
        # マッピングルール
        mapping = {
            '正番': ['正番', '馬番', 'UMABAN'], # 最優先
            '場名': ['場所', '場名', '会場'],
            'R': ['R', 'レース', '番組', 'Ｒ'],
            '馬名': ['馬名', '名称'],
            '単ｵｯｽﾞ': ['単勝', 'オッズ', '単ｵｯｽﾞ'],
            '騎手': ['騎手'], '厩舎': ['厩舎', '調教'], '馬主': ['馬主'],
            '着順': ['着順', '着', '結果', '順位']
        }
        
        # 1. 重要な「正番」をまず探す（逆や枠が含まれるものは拒否）
        for idx, h_name in enumerate(headers):
            if any(k in h_name for k in ['正番', '馬番']) and not any(k in h_name for k in ['逆', '枠']):
                col_idx['正番'] = idx
                break
        
        # 2. それでも見つからなければ、F列（5）をチェック（名前が「逆」でなければ採用）
        if '正番' not in col_idx and len(df_data.columns) >= 6:
            if not any(k in headers[5] for k in ['逆', '枠']):
                col_idx['正番'] = 5

        # 3. その他の列を名寄せ
        for internal, keys in mapping.items():
            if internal in col_idx: continue
            for idx, h_name in enumerate(headers):
                if any(k in h_name for k in keys):
                    col_idx[internal] = idx; break

        # 番号に基づいて新しい表を作る
        df = pd.DataFrame(index=df_data.index)
        detected_info = []
        for internal, idx in col_idx.items():
            if idx is not None and idx < len(df_data.columns):
                df[internal] = df_data.iloc[:, idx]
                detected_info.append(f"{internal}: {headers[idx]}({idx+1}列目)")
        
        st.sidebar.success("✅ 列の特定成功: " + " / ".join(detected_info))
        
        # 必須項目補完
        for col in ['R', '正番', '場名', '馬名', '単ｵｯｽﾞ', '着順']:
            if col not in df.columns:
                df[col] = np.nan if col == '着順' else (0 if col in ['R','正番'] else (99.0 if col == '単ｵｯｽﾞ' else "不明"))

        def clean_num(val):
            s = to_half_width(val)
            match = re.search(r'(\d+\.?\d*)', s)
            return float(match.group(1)) if match else 0.0

        df['R'] = df['R'].apply(clean_num).astype(int)
        df['正番'] = df['正番'].apply(clean_num).astype(int)
        df['単ｵｯｽﾞ'] = df['単ｵｯｽﾞ'].apply(clean_num).replace(0, 99.0)
        df['場名'] = df['場名'].astype(str).str.strip().replace('nan', '不明')
        
        df = df[df['R'] > 0].copy()
        
        # --- 配置判定 ---
        df['青塗フラグ'] = 0; df['判定'] = ""
        df['頭数'] = df.groupby(['場名', 'R'])['正番'].transform('max').fillna(16)
        df['逆番'] = (df['頭数'] + 1) - df['正番']
        df['正循環'] = df['頭数'] + df['正番']
        df['逆循環'] = df['頭数'] + df['逆番']
        
        idx_map = {(row['場名'], row['R'], int(row['正番'])): i for i, row in df.iterrows()}
        for col in ['騎手', '厩舎', '馬主']:
            if col not in df.columns: continue
            for name, group in df.groupby(['場名', col] if col=='騎手' else col):
                if len(group) < 2 or str(name) in ['nan', '', '不明']: continue
                all_sets = [{r['正番'], r['逆番'], r['正循環'], r['逆循環']} for _, r in group.iterrows()]
                common = set.intersection(*all_sets)
                if common:
                    for _, row in group.iterrows():
                        key = (row['場名'], row['R'], int(row['正番']))
                        if key in idx_map:
                            df.at[idx_map[key], '青塗フラグ'] = 1
                            df.at[idx_map[key], '判定'] += f"★{col}青塗 "

        # AI激走確率
        df['AI激走確率'] = 0.0
        if model:
            try:
                X = df[['単ｵｯｽﾞ', '青塗フラグ']].copy()
                X['ペアフラグ'] = 0; X['前日フラグ'] = 0
                probs = model.predict_proba(X)
                df['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
            except: pass

        return df, "success"
    except Exception as e:
        return pd.DataFrame(), traceback.format_exc()

# --- 3. UI ---
st.title("🏇 AI配置分析：正番誤認防止版")

up_file = st.sidebar.file_uploader("当日配置表", type=['xlsx', 'csv'])

if up_file:
    if 'df' not in st.session_state or st.sidebar.button("🔄 データを再解析"):
        res, status = bulletproof_load(up_file)
        if status == "success":
            st.session_state['df'] = res
        else:
            st.error("解析失敗"); st.code(status)

    if 'df' in st.session_state:
        df = st.session_state['df'].copy()
        df['当日バイアス'] = 0.0
        if '着順' in df.columns:
            df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
            hits = df[df['着順'] <= 3]
            if not hits.empty:
                bias = (len(hits[hits['青塗フラグ'] == 1]) / len(hits)) * 15.0
                df.loc[df['青塗フラグ'] == 1, '当日バイアス'] = bias

        df['期待値'] = df['AI激走確率'] + df['当日バイアス']
        places = sorted([p for p in df['場名'].unique() if str(p) not in ['nan','不明']])
        
        if places:
            target_p = st.sidebar.selectbox("会場を選択", places)
            r_list = sorted(df[df['場名'] == target_p]['R'].unique())
            target_r = st.sidebar.selectbox("レースを選択", r_list)

            view = df[(df['場名'] == target_p) & (df['R'] == target_r)].sort_values('期待値', ascending=False)
            st.subheader(f"📊 {target_p} {int(target_r)}R 激走予測")
            
            show_cols = ['正番', '馬名', '単ｵｯｽﾞ', 'AI激走確率', '当日バイアス', '期待値', '判定', '着順']
            final_cols = [c for c in show_cols if c in view.columns]
            
            ed = st.data_editor(view[final_cols], key=f"ed_{target_p}_{target_r}", 
                                hide_index=True, use_container_width=True,
                                column_config={
                                    "期待値": st.column_config.ProgressColumn("最終期待値", format="%.1f", min_value=0, max_value=100),
                                    "正番": st.column_config.NumberColumn("馬番", format="%d")
                                })
            
            if st.button("🔄 着順を保存して午後の流れを反映"):
                for _, row in ed.iterrows():
                    st.session_state['df'].loc[(st.session_state['df']['場名']==target_p) & 
                                               (st.session_state['df']['R']==target_r) & 
                                               (st.session_state['df']['正番']==row['正番']), '着順'] = row['着順']
                st.rerun()
