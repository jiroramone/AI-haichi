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

# --- 2. データ読み込み（物理インデックス方式） ---
def bulletproof_load(file):
    try:
        # ヘッダーなしで全データを読み込む
        if file.name.endswith('.xlsx'):
            df_raw = pd.read_excel(file, header=None, engine='openpyxl')
        else:
            try: df_raw = pd.read_csv(file, header=None, encoding='utf-8')
            except: df_raw = pd.read_csv(file, header=None, encoding='cp932')
        
        # 見出し行（ヘッダー）を特定
        best_row, max_hits = 0, 0
        keywords = ['場所', 'R', '馬名', 'オッズ']
        for i in range(min(len(df_raw), 30)):
            row_vals = [str(x) for x in df_raw.iloc[i].values]
            hits = sum(1 for k in keywords if any(k in v for v in row_vals))
            if hits > max_hits:
                max_hits, best_row = hits, i
        
        # データの切り出し
        header_names = [str(x).strip() for x in df_raw.iloc[best_row].values]
        df = df_raw.iloc[best_row+1:].reset_index(drop=True)
        
        # --- 物理位置による列特定 ---
        # F列（インデックス5）は強制的に正番
        col_indices = {'正番': 5 if len(df.columns) > 5 else None}
        
        # その他の列をキーワードで探索（インデックスを保存）
        mapping_rules = {
            '場名': ['場所', '場名', '会場'],
            'R': ['R', 'レース', '番組'],
            '馬名': ['馬名', '名称'],
            '単ｵｯｽﾞ': ['単勝', 'オッズ', '単ｵｯｽﾞ'],
            '騎手': ['騎手'], '厩舎': ['厩舎', '調教'], '馬主': ['馬主', 'オーナー'],
            '着順': ['着順', '着', '結果']
        }
        
        for internal, keys in mapping_rules.items():
            if internal in col_indices and col_indices[internal] is not None: continue
            for idx, name in enumerate(header_names):
                if any(k in name for k in keys):
                    col_indices[internal] = idx
                    break

        # 新しいデータフレームを作成（名前の重複エラーを避けるため）
        new_df = pd.DataFrame()
        for internal, idx in col_indices.items():
            if idx is not None and idx < len(df.columns):
                new_df[internal] = df.iloc[:, idx]
            else:
                new_df[internal] = np.nan if internal == '着順' else (0 if internal in ['R','正番'] else "")

        # 数値クリーニング
        def clean_num(val):
            s = to_half_width(val)
            match = re.search(r'(\d+\.?\d*)', str(s))
            return float(match.group(1)) if match else 0.0

        new_df['R'] = new_df['R'].apply(clean_num).astype(int)
        new_df['正番'] = new_df['正番'].apply(clean_num).astype(int)
        new_df['単ｵｯｽﾞ'] = new_df['単ｵｯｽﾞ'].apply(clean_num).replace(0, 99.0)
        
        # 無効な行を除去
        new_df = new_df[new_df['R'] > 0].copy()
        
        # --- 配置判定（青塗） ---
        new_df['青塗フラグ'] = 0; new_df['判定'] = ""
        # 逆番などの計算
        new_df['頭数'] = new_df.groupby(['場名', 'R'])['正番'].transform('max').fillna(16)
        new_df['逆番'] = (new_df['頭数'] + 1) - new_df['正番']
        new_df['正循環'] = new_df['頭数'] + new_df['正番']
        new_df['逆循環'] = new_df['頭数'] + new_df['逆番']
        
        idx_map = {(row['場名'], row['R'], int(row['正番'])): i for i, row in new_df.iterrows()}
        
        for col in ['騎手', '厩舎', '馬主']:
            for name, group in new_df.groupby(['場名', col] if col=='騎手' else col):
                if len(group) < 2 or str(name) in ['nan', '', '不明']: continue
                all_sets = [{r['正番'], r['逆番'], r['正循環'], r['逆循環']} for _, r in group.iterrows()]
                common = set.intersection(*all_sets)
                if common:
                    for _, row in group.iterrows():
                        key = (row['場名'], row['R'], int(row['正番']))
                        if key in idx_map:
                            new_df.at[idx_map[key], '青塗フラグ'] = 1
                            new_df.at[idx_map[key], '判定'] += f"★{col}青塗 "

        # AI予測
        if model:
            try:
                X = new_df[['単ｵｯｽﾞ', '青塗フラグ']].copy()
                X['ペアフラグ'] = 0; X['前日フラグ'] = 0
                probs = model.predict_proba(X)
                new_df['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
            except: new_df['AI激走確率'] = 0.0
        else: new_df['AI激走確率'] = 0.0

        return new_df, "success"
    except Exception as e:
        return pd.DataFrame(), str(e)

# --- 3. UI ---
st.title("🏇 AI配置分析：最終安定版システム")

up_file = st.sidebar.file_uploader("当日配置表", type=['xlsx', 'csv'])

if up_file:
    if 'df' not in st.session_state or st.sidebar.button("🔄 再解析"):
        res, status = bulletproof_load(up_file)
        if status == "success": st.session_state['df'] = res
        else: st.error(f"解析失敗: {status}")

    if 'df' in st.session_state:
        df = st.session_state['df'].copy()
        
        # 流動的補正（結果入力がある場合）
        df['当日バイアス'] = 0.0
        if '着順' in df.columns:
            df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
            hits = df[df['着順'] <= 3]
            if not hits.empty:
                bias = (len(hits[hits['青塗フラグ'] == 1]) / len(hits)) * 10.0
                df.loc[df['青塗フラグ'] == 1, '当日バイアス'] = bias

        df['期待値'] = df['AI激走確率'] + df['当日バイアス']
        
        places = sorted([p for p in df['場名'].unique() if str(p) not in ['nan','None','不明']])
        
        if places:
            target_p = st.sidebar.selectbox("会場", places)
            r_list = sorted(df[df['場名'] == target_p]['R'].unique())
            target_r = st.sidebar.selectbox("レース", r_list)

            view = df[(df['場名'] == target_p) & (df['R'] == target_r)].sort_values('期待値', ascending=False)
            
            st.subheader(f"📊 {target_p} {int(target_r)}R 予測")
            
            # テーブル表示（KeyErrorを絶対に防ぐ）
            show_cols = ['正番', '馬名', '単ｵｯｽﾞ', 'AI激走確率', '当日バイアス', '期待値', '判定', '着順']
            final_cols = [c for c in show_cols if c in view.columns]
            
            ed = st.data_editor(view[final_cols], key=f"ed_{target_p}_{target_r}", 
                                hide_index=True, use_container_width=True,
                                column_config={"期待値": st.column_config.ProgressColumn(min_value=0, max_value=100)})
            
            if st.button("🔄 着順を保存してバイアス更新"):
                for _, row in ed.iterrows():
                    st.session_state['df'].loc[(st.session_state['df']['場名']==target_p) & 
                                               (st.session_state['df']['R']==target_r) & 
                                               (st.session_state['df']['正番']==row['正番']), '着順'] = row['着順']
                st.rerun()
        else:
            st.error("会場を特定できませんでした。")
else:
    st.info("左からファイルをアップロードしてください。")
