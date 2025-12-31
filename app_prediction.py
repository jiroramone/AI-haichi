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

# --- 2. データ読み取り（絶対表示させるマン） ---
def safe_load_and_analyze(file):
    try:
        # ファイル読み込み
        if file.name.endswith('.xlsx'):
            df_raw = pd.read_excel(file, header=None, engine='openpyxl')
        else:
            try: df_raw = pd.read_csv(file, header=None, encoding='utf-8')
            except: df_raw = pd.read_csv(file, header=None, encoding='cp932')
        
        # 見出し行を自動探索（点数制）
        best_row, max_hits = 0, 0
        keywords = ['場所', 'R', '馬名', 'オッズ']
        for i in range(min(len(df_raw), 25)):
            row_str = "".join(df_raw.iloc[i].astype(str))
            hits = sum(1 for k in keywords if k in row_str)
            if hits > max_hits:
                max_hits, best_row = hits, i
        
        df = df_raw.iloc[best_row:].reset_index(drop=True)
        headers = [str(c).strip() for c in df.iloc[0]]
        
        # 【指示】F列(6列目)を強制的に正番に固定
        if len(headers) >= 6:
            headers[5] = "正番"
        
        df.columns = headers
        df = df.iloc[1:].reset_index(drop=True)

        # 項目名の名寄せ（「場所」を最優先で「場名」へ）
        mapping = {
            '場名': ['場所', '場名', '会場'],
            'R': ['R', 'レース', '番組'],
            '馬名': ['馬名', '名称'],
            '単ｵｯｽﾞ': ['単ｵｯｽﾞ', 'オッズ', '単勝'],
            '着順': ['着順', '着', '結果'],
            '騎手': ['騎手'], '厩舎': ['厩舎', '調教師'], '馬主': ['馬主']
        }
        col_map = {}
        for internal, keys in mapping.items():
            for c in df.columns:
                if any(k in str(c) for k in keys):
                    col_map[c] = internal; break
        df = df.rename(columns=col_map)

        # 数値のクリーンアップ
        def extract_num(val):
            s = to_half_width(val)
            match = re.search(r'(\d+\.?\d*)', str(s))
            return float(match.group(1)) if match else 0.0

        for c in ['R', '正番', '単ｵｯｽﾞ']:
            if c in df.columns:
                df[c] = df[c].apply(extract_num)
            else:
                df[c] = 0.0 if c != '単ｵｯｽﾞ' else 99.0

        # 配置判定用の基本計算
        df = df[df['R'] > 0].copy()
        df['青塗フラグ'] = 0; df['当日バイアス'] = 0.0; df['判定'] = ""
        df['頭数'] = df.groupby(['場名', 'R'])['正番'].transform('max').fillna(16)
        df['逆番'] = (df['頭数'] + 1) - df['正番']
        df['正循環'] = df['頭数'] + df['正番']
        df['逆循環'] = df['頭数'] + df['逆番']
        
        idx_map = {(row['場名'], row['R'], int(row['正番'])): idx for idx, row in df.iterrows()}

        # 青塗判定（配置の不変ルール）
        for col in ['騎手', '厩舎', '馬主']:
            if col not in df.columns: continue
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

        # AI基礎確率（model.pklが期待する4つの特徴量）
        if model:
            X = df[['単ｵｯｽﾞ', '青塗フラグ']].copy()
            X['ペアフラグ'] = 0; X['前日フラグ'] = 0 # 配置ルール拡張用
            probs = model.predict_proba(X)
            df['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
        else:
            df['AI激走確率'] = 0.0

        return df, "success"
    except Exception as e:
        return pd.DataFrame(), str(e)

# --- 3. UI画面 ---
st.title("🎯 AI配置馬券 激走予測システム（流動解析版）")

up_file = st.sidebar.file_uploader("当日配置表をアップロード", type=['xlsx', 'csv'])

if up_file:
    # データ読み込み・解析（session_stateで保持）
    if 'df' not in st.session_state or st.sidebar.button("🔄 データを再解析"):
        with st.spinner('解析中...'):
            res_df, status = safe_load_and_analyze(up_file)
            if status == "success":
                st.session_state['df'] = res_df
            else:
                st.error(f"解析失敗: {status}")

    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # 【流動的ロジック】結果によるバイアスの計算
        if '着順' in df.columns:
            df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
            hits = df[df['着順'] <= 3]
            if not hits.empty:
                # 青塗が来ている割合を計算
                bias = (len(hits[hits['青塗フラグ'] == 1]) / len(hits)) * 10.0
                df.loc[df['青塗フラグ'] == 1, '当日バイアス'] = bias

        df['最終期待値'] = df['AI激走確率'] + df['当日バイアス']
        
        # 会場とレースの選択
        places = sorted([p for p in df['場名'].unique() if str(p) != 'nan' and p != '不明'])
        
        if places:
            st.sidebar.divider()
            target_p = st.sidebar.selectbox("会場を選択", places)
            r_list = sorted(df[df['場名'] == target_p]['R'].unique())
            target_r = st.sidebar.selectbox("レースを選択", r_list)

            # 表示
            view = df[(df['場名'] == target_p) & (df['R'] == target_r)].sort_values('最終期待値', ascending=False)
            
            st.subheader(f"📊 {target_p} {int(target_r)}R 予測結果")
            
            # 推奨馬カード
            top = view.iloc[0]
            st.info(f"👑 AI最推奨: {int(top['正番'])}番 {top['馬名']} (期待値 {top['最終期待値']:.1f})")

            # メインテーブル
            # 必要な列だけ選んで表示（KeyErrorを避ける）
            cols_to_show = ['正番', '馬名', '単ｵｯｽﾞ', 'AI激走確率', '当日バイアス', '最終期待値', '判定', '着順']
            available_cols = [c for c in cols_to_show if c in view.columns]
            
            ed = st.data_editor(view[available_cols], 
                                key=f"ed_{target_p}_{target_r}", 
                                hide_index=True, use_container_width=True,
                                column_config={
                                    "最終期待値": st.column_config.ProgressColumn(min_value=0, max_value=100),
                                    "正番": st.column_config.NumberColumn(format="%d")
                                })
            
            # 結果入力の反映ボタン
            if st.button("🔄 入力した着順を保存してバイアスを更新"):
                for _, row in ed.iterrows():
                    st.session_state['df'].loc[(st.session_state['df']['場名']==target_p) & (st.session_state['df']['R']==target_r) & (st.session_state['df']['正番']==row['正番']), '着順'] = row['着順']
                st.rerun()

            st.divider()
            with st.expander("🔍 今日の配置傾向分析"):
                hits_summary = df[df['着順'] <= 3]
                if not hits_summary.empty:
                    st.write("### 的中したパターンの内訳")
                    st.plotly_chart(px.pie(hits_summary['判定'].value_counts().reset_index(), values='count', names='index', hole=0.4))
                else:
                    st.write("着順を入力すると、今日の配置の流れが表示されます。")
        else:
            st.error("会場名が特定できません。エクセルの1行目（場所・場名など）を確認してください。")
