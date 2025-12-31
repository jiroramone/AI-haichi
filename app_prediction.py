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
    if pd.isna(text): return text
    table = str.maketrans('０１２３４５６７８９．', '0123456789.')
    return re.sub(r'[^\d\.]', '', str(text).translate(table))

# --- 2. データ読み込み（名前ではなく番号で列を特定） ---
def bulletproof_load(file):
    try:
        # ヘッダーなしで全データを読み込む（名前の混乱を防ぐ）
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
        
        # データの切り出し
        headers = [str(x).strip() for x in df_raw.iloc[best_row].values]
        df_data = df_raw.iloc[best_row+1:].reset_index(drop=True)
        
        # --- 列の「番号」による特定ロジック ---
        col_indices = {}
        
        # ① F列（インデックス5）は強制的に「正番」
        if len(df_data.columns) > 5:
            col_indices['正番'] = 5
        
        # ② 他の列をキーワードから「番号」で特定
        mapping_rules = {
            '場名': ['場所', '場名', '会場'],
            'R': ['R', 'レース', '番組'],
            '馬名': ['馬名', '名称'],
            '単ｵｯｽﾞ': ['単勝', 'オッズ', '単ｵｯｽﾞ'],
            '着順': ['着順', '着', '結果', '順位'],
            '騎手': ['騎手'], '厩舎': ['厩舎', '調教'], '馬主': ['馬主']
        }
        
        for internal, keys in mapping_rules.items():
            if internal in col_indices: continue # すでに決まっていれば飛ばす
            for idx, h_name in enumerate(headers):
                if any(k in h_name for k in keys):
                    col_indices[internal] = idx; break

        # 番号に基づいて新しいクリーンなデータフレームを再構築
        df = pd.DataFrame()
        for internal, idx in col_indices.items():
            if idx is not None and idx < len(df_data.columns):
                df[internal] = df_data.iloc[:, idx]
        
        # 必須列が足りない場合の補完（エラー防止）
        for col in ['R', '正番', '場名', '馬名', '単ｵｯｽﾞ', '着順']:
            if col not in df.columns:
                df[col] = np.nan if col == '着順' else (0 if col in ['R','正番'] else (99.0 if col == '単ｵｯｽﾞ' else "不明"))

        # 数値クリーニング
        def clean_num(val):
            s = to_half_width(val)
            match = re.search(r'(\d+\.?\d*)', str(s))
            return float(match.group(1)) if match else 0.0

        df['R'] = df['R'].apply(clean_num).astype(int)
        df['正番'] = df['正番'].apply(clean_num).astype(int)
        df['単ｵｯｽﾞ'] = df['単ｵｯｽﾞ'].apply(clean_num).replace(0, 99.0)
        
        # --- 配置判定（青塗） ---
        df = df[df['R'] > 0].copy()
        df['青塗フラグ'] = 0; df['判定'] = ""
        df['頭数'] = df.groupby(['場名', 'R'])['正番'].transform('max').fillna(16)
        df['逆番'] = (df['頭数'] + 1) - df['正番']
        df['正循環'] = df['頭数'] + df['正番']
        df['逆循環'] = df['頭数'] + df['逆番']
        
        idx_map = {(row['場名'], row['R'], int(row['正番'])): i for i, row in df.iterrows()}
        for col in ['騎手', '厩舎', '馬主']:
            if col in df.columns:
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

        # AI基礎予測
        if model:
            try:
                X = df[['単ｵｯｽﾞ', '青塗フラグ']].copy()
                X['ペアフラグ'] = 0; X['前日フラグ'] = 0
                probs = model.predict_proba(X)
                df['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
            except: df['AI激走確率'] = 0.0
        else: df['AI激走確率'] = 0.0

        return df, "success"
    except Exception as e:
        return pd.DataFrame(), str(e)

# --- 3. UI ---
st.title("🎯 AI配置分析：流動バイアス・エラー根絶版")

up_file = st.sidebar.file_uploader("当日配置表(Excel/CSV)", type=['xlsx', 'csv'])

if up_file:
    if 'df' not in st.session_state or st.sidebar.button("🔄 データを再読み込み"):
        res, status = bulletproof_load(up_file)
        if status == "success": st.session_state['df'] = res
        else: st.error(f"解析失敗: {status}")

    if 'df' in st.session_state:
        df = st.session_state['df'].copy()
        
        # 【流動的ロジック】結果によるバイアス補正
        df['当日バイアス'] = 0.0
        if '着順' in df.columns:
            df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
            hits = df[df['着順'] <= 3]
            if not hits.empty:
                # 「今日の青塗の的中率」を計算してボーナス値を決定
                bias = (len(hits[hits['青塗フラグ'] == 1]) / len(hits)) * 15.0
                df.loc[df['青塗フラグ'] == 1, '当日バイアス'] = bias

        df['期待値'] = df['AI激走確率'] + df['当日バイアス']
        
        places = sorted([p for p in df['場名'].unique() if str(p) not in ['nan','None','不明']])
        
        if places:
            target_p = st.sidebar.selectbox("会場を選択", places)
            r_list = sorted(df[df['場名'] == target_p]['R'].unique())
            target_r = st.sidebar.selectbox("レースを選択", r_list)

            view = df[(df['場名'] == target_p) & (df['R'] == target_r)].sort_values('期待値', ascending=False)
            st.subheader(f"📊 {target_p} {int(target_r)}R 激走予測")
            
            # 安全に表示列を構成
            disp_cols = ['正番', '馬名', '単ｵｯｽﾞ', 'AI激走確率', '当日バイアス', '期待値', '判定', '着順']
            final_cols = [c for c in disp_cols if c in view.columns]
            
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
        else:
            st.error("会場を特定できませんでした。")
