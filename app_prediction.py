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

# --- 2. データ読み込み（エラー回避・F列固定） ---
def safe_load_and_analyze(file):
    try:
        if file.name.endswith('.xlsx'):
            df_raw = pd.read_excel(file, header=None, engine='openpyxl')
        else:
            try: df_raw = pd.read_csv(file, header=None, encoding='utf-8')
            except: df_raw = pd.read_csv(file, header=None, encoding='cp932')
        
        # ヘッダー行の自動探索
        best_row, max_hits = 0, 0
        keywords = ['場所', 'R', '馬名', 'オッズ']
        for i in range(min(len(df_raw), 30)):
            row_vals = [str(x) for x in df_raw.iloc[i].values]
            hits = sum(1 for k in keywords if any(k in v for v in row_vals))
            if hits > max_hits:
                max_hits, best_row = hits, i
        
        df = df_raw.iloc[best_row:].reset_index(drop=True)
        # 列名リストを一旦作成
        headers = [str(c).strip() for c in df.iloc[0].values]
        
        # 【重要】F列(6列目)を強制的に正番に固定
        if len(headers) >= 6:
            headers[5] = "正番"
        
        df.columns = headers
        df = df.iloc[1:].reset_index(drop=True)

        # 項目名の名寄せ（エラー回避のため辞書で一括変換）
        name_map = {}
        mapping_rules = {
            '場名': ['場所', '場名', '会場', '競馬場'],
            'R': ['R', 'レース', '番組', 'Ｒ'],
            '馬名': ['馬名', '名称', '馬'],
            '単ｵｯｽﾞ': ['単ｵｯｽﾞ', 'オッズ', '単勝', '単オッズ'],
            '着順': ['着順', '着', '結果', '順位'],
            '騎手': ['騎手'], '厩舎': ['厩舎', '調教師'], '馬主': ['馬主']
        }
        
        # 現在の列名（headers）をループして、ルールに合致するものを探す
        for col in df.columns:
            col_str = str(col)
            for internal, keywords in mapping_rules.items():
                if any(k in col_str for k in keywords):
                    name_map[col] = internal
                    break
        
        df = df.rename(columns=name_map)
        
        # 重複列名の解消
        cols = []
        counts = {}
        for c in df.columns:
            c_str = str(c)
            if c_str in counts:
                counts[c_str] += 1
                cols.append(f"{c_str}_{counts[c_str]}")
            else:
                counts[c_str] = 0
                cols.append(c_str)
        df.columns = cols

        # 数値クリーニング
        def extract_num(val):
            s = to_half_width(val)
            match = re.search(r'(\d+\.?\d*)', str(s))
            return float(match.group(1)) if match else 0.0

        for c in ['R', '正番', '単ｵｯｽﾞ']:
            if c in df.columns:
                df[c] = df[c].apply(extract_num)
            else:
                df[c] = 0.0 if c != '単ｵｯｽﾞ' else 99.0

        df = df[df['R'] > 0].copy()
        df['青塗フラグ'] = 0; df['当日バイアス'] = 0.0; df['判定'] = ""
        
        # 配置計算の基本（頭数・逆番など）
        df['頭数'] = df.groupby(['場名', 'R'])['正番'].transform('max').fillna(16)
        df['逆番'] = (df['頭数'] + 1) - df['正番']
        df['正循環'] = df['頭数'] + df['正番']
        df['逆循環'] = df['頭数'] + df['逆番']
        
        # --- 配置判定（青塗ロジック） ---
        idx_map = {(row['場名'], row['R'], int(row['正番'])): idx for idx, row in df.iterrows()}
        for col in ['騎手', '厩舎', '馬主']:
            if col in df.columns:
                for name, group in df.groupby(['場名', col] if col=='騎手' else col):
                    if len(group) < 2 or str(name) in ['nan', '', '不明']: continue
                    all_sets = [{r['正番'], r['逆番'], r['正循環'], r['逆循環']} for _, r in group.iterrows()]
                    common = set.intersection(*all_sets)
                    if common:
                        for _, row in group.iterrows():
                            idx = idx_map.get((row['場名'], row['R'], int(row['正番'])))
                            if idx is not None:
                                df.at[idx, '青塗フラグ'] = 1
                                df.at[idx, '判定'] += f"★{col}青塗 "

        # --- AI基礎確率 ---
        if model:
            try:
                X = df[['単ｵｯｽﾞ', '青塗フラグ']].copy()
                X['ペアフラグ'] = 0; X['前日フラグ'] = 0
                probs = model.predict_proba(X)
                df['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
            except: df['AI激走確率'] = 0.0
        else:
            df['AI激走確率'] = 0.0

        return df, "success"
    except Exception as e:
        return pd.DataFrame(), str(e)

# --- 3. UI画面 ---
st.title("🏇 AI配置分析：流動・最適化システム")

up_file = st.sidebar.file_uploader("当日配置表(Excel/CSV)", type=['xlsx', 'csv'])

if up_file:
    # データの初回解析
    if 'df' not in st.session_state or st.sidebar.button("🔄 データを再解析"):
        res_df, status = safe_load_and_analyze(up_file)
        if status == "success":
            st.session_state['df'] = res_df
        else:
            st.error(f"解析失敗: {status}")

    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # 【流動的ロジック】結果によるバイアス補正
        if '着順' in df.columns:
            df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
            hits = df[df['着順'] <= 3]
            if not hits.empty:
                # 的中馬における「青塗」の割合から、今日の重要度を算出
                bias = (len(hits[hits['青塗フラグ'] == 1]) / len(hits)) * 10.0
                df.loc[df['青塗フラグ'] == 1, '当日バイアス'] = bias

        df['期待値'] = df['AI激走確率'] + df['当日バイアス']
        
        # 会場選択（「場名」は「場所」から変換済み）
        places = sorted([p for p in df['場名'].unique() if str(p) not in ['nan', '不明']])
        
        if places:
            target_p = st.sidebar.selectbox("会場を選択", places)
            r_list = sorted(df[df['場名'] == target_p]['R'].unique())
            target_r = st.sidebar.selectbox("レースを選択", r_list)

            # 表示用データの抽出
            view = df[(df['場名'] == target_p) & (df['R'] == target_r)].sort_values('期待値', ascending=False)
            
            st.subheader(f"📊 {target_p} {int(target_r)}R 予測結果")
            
            # メイン表示
            disp_cols = ['正番', '馬名', '単ｵｯｽﾞ', 'AI激走確率', '当日バイアス', '期待値', '判定', '着順']
            available_cols = [c for c in disp_cols if c in view.columns]
            
            ed = st.data_editor(view[available_cols], 
                                key=f"ed_{target_p}_{target_r}", 
                                hide_index=True, use_container_width=True,
                                column_config={
                                    "期待値": st.column_config.ProgressColumn("最終期待値", format="%.1f", min_value=0, max_value=100),
                                    "正番": st.column_config.NumberColumn("馬番", format="%d")
                                })
            
            if st.button("🔄 着順を確定して今日のバイアスを更新"):
                # 入力された着順を元のデータフレームに反映
                for _, row in ed.iterrows():
                    st.session_state['df'].loc[(st.session_state['df']['場名']==target_p) & 
                                               (st.session_state['df']['R']==target_r) & 
                                               (st.session_state['df']['正番']==row['正番']), '着順'] = row['着順']
                st.rerun()
        else:
            st.error("会場（場所）が特定できません。データ形式を確認してください。")

else:
    st.info("左側のメニューからファイルをアップロードしてください。")
