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
        except Exception as e:
            st.sidebar.error(f"モデル読込エラー: {e}")
            return None
    return None

model = load_ai_model()

def to_half_width(text):
    if pd.isna(text): return text
    table = str.maketrans('０１２３４５６７８９．', '0123456789.')
    return re.sub(r'[^\d\.]', '', str(text).translate(table))

# --- 2. データ解析エンジン（期待値計算を強化） ---
def bulletproof_analyze(file):
    try:
        # ファイル読み込み
        if file.name.endswith('.xlsx'):
            df_raw = pd.read_excel(file, header=None, engine='openpyxl')
        else:
            try: df_raw = pd.read_csv(file, header=None, encoding='utf-8')
            except: df_raw = pd.read_csv(file, header=None, encoding='cp932')
        
        # ヘッダー行を自動探索
        best_row, max_hits = 0, 0
        keywords = ['場所', 'R', '馬名', 'オッズ']
        for i in range(min(len(df_raw), 30)):
            row_vals = [str(x) for x in df_raw.iloc[i].values]
            hits = sum(1 for k in keywords if any(k in v for v in row_vals))
            if hits > max_hits:
                max_hits, best_row = hits, i
        
        headers = [str(x).strip() for x in df_raw.iloc[best_row].values]
        df = df_raw.iloc[best_row+1:].reset_index(drop=True)
        
        # 【重要】F列（6列目）を強制的に「正番」に固定
        if len(df.columns) >= 6:
            headers[5] = "正番"
        
        df.columns = headers

        # 列名の名寄せ
        mapping = {
            '場名': ['場所', '場名', '会場', '競馬場'],
            'R': ['R', 'レース', '番組', 'Ｒ'],
            '馬名': ['馬名', '名称'],
            '単ｵｯｽﾞ': ['単勝', 'オッズ', '単ｵｯｽﾞ'],
            '着順': ['着順', '着', '結果', '順位'],
            '騎手': ['騎手'], '厩舎': ['厩舎', '調教師'], '馬主': ['馬主']
        }
        
        col_map = {}
        for internal, keys in mapping.items():
            for c in df.columns:
                if any(k in str(c) for k in keys):
                    col_map[c] = internal; break
        df = df.rename(columns=col_map)

        # 数値クリーニング
        def clean_num(val):
            s = to_half_width(val)
            match = re.search(r'(\d+\.?\d*)', str(s))
            return float(match.group(1)) if match else 0.0

        for c in ['R', '正番', '単ｵｯｽﾞ']:
            if c in df.columns: df[c] = df[c].apply(clean_num)
            else: df[c] = 0.0 if c != '単ｵｯｽﾞ' else 99.0

        df = df[df['R'] > 0].copy()
        
        # --- 配置判定（青塗ロジック） ---
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

        # --- AI激走期待値の計算 ---
        df['AI激走確率'] = 0.0
        if model:
            try:
                X = df[['単ｵｯｽﾞ', '青塗フラグ']].copy()
                X['ペアフラグ'] = 0; X['前日フラグ'] = 0
                probs = model.predict_proba(X)
                df['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
            except Exception as e:
                st.sidebar.warning(f"AI計算中: {e}")

        df['当日バイアス'] = 0.0
        df['期待値'] = df['AI激走確率'] # 初回はAI確率をそのまま入れる
        
        if '着順' not in df.columns: df['着順'] = np.nan
        
        return df, "success"
    except Exception as e:
        return pd.DataFrame(), str(e)

# --- 3. UI画面 ---
st.title("🏇 AI配置分析：期待値・流動解析システム")



up_file = st.sidebar.file_uploader("当日配置表(Excel/CSV)", type=['xlsx', 'csv'])

if up_file:
    if 'df' not in st.session_state or st.sidebar.button("🔄 データを再読み込み"):
        res, status = bulletproof_analyze(up_file)
        if status == "success":
            st.session_state['df'] = res
        else:
            st.error(f"解析失敗: {status}")

    if 'df' in st.session_state:
        # 流動的な期待値更新
        df = st.session_state['df'].copy()
        
        # 着順が入力されている馬をチェックしてバイアス計算
        df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
        hits = df[df['着順'] <= 3]
        if not hits.empty:
            # 青塗パターンの的中率からバイアス（ボーナス点）を算出
            bias_rate = (len(hits[hits['青塗フラグ'] == 1]) / len(hits)) * 20.0
            df.loc[df['青塗フラグ'] == 1, '当日バイアス'] = bias_rate
        
        # 期待値 = AIの基本予測 + 今日のバイアス
        df['期待値'] = df['AI激走確率'] + df['当日バイアス']
        
        places = sorted([p for p in df['場名'].unique() if str(p) not in ['nan','不明']])
        
        if places:
            target_p = st.sidebar.selectbox("会場を選択", places)
            r_list = sorted(df[df['場名'] == target_p]['R'].unique())
            target_r = st.sidebar.selectbox("レースを選択", r_list)

            view = df[(df['場名'] == target_p) & (df['R'] == target_r)].sort_values('期待値', ascending=False)
            
            st.subheader(f"📊 {target_p} {int(target_r)}R 激走予測")
            
            # 推奨馬を大きく表示（期待値が出ているか確認用）
            top_horse = view.iloc[0]
            st.info(f"👑 AI最推奨: {int(top_horse['正番'])}番 {top_horse['馬名']} (期待値: {top_horse['期待値']:.1f})")

            # メインテーブル
            disp_cols = ['正番', '馬名', '単ｵｯｽﾞ', 'AI激走確率', '当日バイアス', '期待値', '判定', '着順']
            available_cols = [c for c in disp_cols if c in view.columns]
            
            ed = st.data_editor(view[available_cols], key=f"ed_{target_p}_{target_r}", 
                                hide_index=True, use_container_width=True,
                                column_config={
                                    "期待値": st.column_config.ProgressColumn("期待値（AI+流れ）", format="%.1f", min_value=0, max_value=100),
                                    "AI激走確率": st.column_config.NumberColumn("AI基礎", format="%.1f%%"),
                                    "正番": st.column_config.NumberColumn("馬番", format="%d")
                                })
            
            if st.button("🔄 入力した着順を保存して午後の期待値を更新"):
                # 入力された「着順」を session_state のメインデータに反映
                for _, row in ed.iterrows():
                    st.session_state['df'].loc[(st.session_state['df']['場名']==target_p) & 
                                               (st.session_state['df']['R']==target_r) & 
                                               (st.session_state['df']['正番']==row['正番']), '着順'] = row['着順']
                st.rerun()
        else:
            st.error("会場（場所）が特定できませんでした。")

else:
    st.info("左側のサイドバーから配置表を読み込んでください。")
