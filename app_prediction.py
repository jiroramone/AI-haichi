import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os
import plotly.express as px

# --- 1. 基本設定 ---
st.set_page_config(page_title="AI配置分析・動的最適化システム", layout="wide")

@st.cache_resource
def load_ai_model():
    if os.path.exists('model.pkl'):
        try:
            with open('model.pkl', 'rb') as f:
                return pickle.load(f)
        except: return None
    return None

model = load_ai_model()

# --- 2. 高精度データ読み込み ---
def load_and_clean_data(file):
    try:
        if file.name.endswith('.xlsx'):
            df_raw = pd.read_excel(file, header=None, engine='openpyxl')
        else:
            try: df_raw = pd.read_csv(file, header=None, encoding='utf-8')
            except: df_raw = pd.read_csv(file, header=None, encoding='cp932')
        
        # ヘッダー探索（キーワードヒット制）
        best_row, max_hits = 0, 0
        keywords = ['場所', 'R', '馬名', '正番']
        for i in range(min(len(df_raw), 25)):
            row_str = "".join(df_raw.iloc[i].astype(str))
            hits = sum(1 for k in keywords if k in row_str)
            if hits > max_hits:
                max_hits, best_row = hits, i
        
        df = df_raw.iloc[best_row:].reset_index(drop=True)
        headers = [str(c).strip() for c in df.iloc[0]]
        
        # 【指示】F列(6列目)を強制的に正番に固定
        if len(headers) >= 6: headers[5] = "正番"
        df.columns = headers
        df = df.iloc[1:].reset_index(drop=True)

        # 列名マッピング
        mapping = {'場名':['場所','場名','会場'],'R':['R','レース'],'正番':['正番','馬番'],'単ｵｯｽﾞ':['単ｵｯｽﾞ','オッズ'],'着順':['着順','着']}
        for internal, keys in mapping.items():
            for c in df.columns:
                if any(k in str(c) for k in keys):
                    df = df.rename(columns={c: internal})
                    break

        # 数値化とクリーンアップ
        for c in ['R', '正番', '単ｵｯｽﾞ']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c].astype(str).str.extract(r'(\d+\.?\d*)', expand=False), errors='coerce').fillna(0)
        
        df = df[df['R'] > 0].copy()
        return df, "success"
    except Exception as e: return pd.DataFrame(), str(e)

# --- 3. 配置解析・AI計算 ---
def run_analysis(df):
    # 配置フラグ初期化
    df['青塗フラグ'] = 0; df['タイプ'] = ""; df['当日バイアス'] = 0.0
    
    # 逆番・循環の計算
    df['頭数'] = df.groupby(['場名', 'R'])['正番'].transform('max')
    df['逆番'] = (df['頭数'] + 1) - df['正番']
    df['正循環'] = df['頭数'] + df['正番']
    df['逆循環'] = df['頭数'] + df['逆番']
    
    idx_map = {(row['場名'], row['R'], int(row['正番'])): idx for idx, row in df.iterrows()}

    # 配置一致（青塗）の判定
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
                        df.at[idx, 'タイプ'] += f"★{col}青塗 "

    # AI基礎確率の算出
    if model:
        X = df[['単ｵｯｽﾞ', '青塗フラグ']].copy()
        X['ペアフラグ'] = 0; X['前日フラグ'] = 0 # 拡張用
        probs = model.predict_proba(X)
        df['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
    else:
        df['AI激走確率'] = 0.0

    return df

# --- 4. 結果を反映する流動的ロジック ---
def update_with_results(df):
    # 着順が入っているレースの結果を分析
    df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
    hit_data = df[df['着順'] <= 3]
    
    if not hit_data.empty:
        # 今日の的中傾向（どの配置が来ているか）をスコア化
        blue_hit_rate = len(hit_data[hit_data['青塗フラグ'] == 1]) / len(df[df['着順'] <= 3])
        # 中盤以降の「青塗」の馬にボーナスを付与
        df.loc[df['青塗フラグ'] == 1, '当日バイアス'] = blue_hit_rate * 10.0
        
    df['最終期待値'] = df['AI激走確率'] + df['当日バイアス']
    return df

# --- 5. UI ---
st.title("🏇 配置馬券術 AI・流動解析システム")

up_curr = st.sidebar.file_uploader("当日配置表", type=['xlsx', 'csv'])

if up_curr:
    if 'main_df' not in st.session_state:
        df_loaded, status = load_and_clean_data(up_curr)
        if status == "success":
            st.session_state['main_df'] = run_analysis(df_loaded)
    
    if 'main_df' in st.session_state:
        # 流動的要素の更新
        full_df = update_with_results(st.session_state['main_df'])
        
        places = sorted([p for p in full_df['場名'].unique() if str(p) != 'nan'])
        
        tab1, tab2 = st.tabs(["📊 激走予測（結果入力）", "📈 的中バイアス分析"])
        
        with tab1:
            target_p = st.selectbox("会場", places)
            p_df = full_df[full_df['場名'] == target_p]
            r_nums = sorted(p_df['R'].unique())
            
            with st.form("edit_form"):
                for r_num in r_nums:
                    with st.expander(f"{int(r_num)}R"):
                        race = p_df[p_df['R'] == r_num].sort_values('最終期待値', ascending=False)
                        # エディタで着順を入力
                        ed = st.data_editor(race[['正番', '馬名', 'AI激走確率', '当日バイアス', '最終期待値', 'タイプ', '着順']], 
                                            key=f"ed_{target_p}_{r_num}", hide_index=True, use_container_width=True,
                                            column_config={"最終期待値": st.column_config.ProgressColumn(min_value=0, max_value=100)})
                        # 入力値をsession_stateへ反映
                        for _, row in ed.iterrows():
                            st.session_state['main_df'].loc[(st.session_state['main_df']['場名']==target_p) & (st.session_state['main_df']['R']==r_num) & (st.session_state['main_df']['正番']==row['正番']), '着順'] = row['着順']
                
                if st.form_submit_button("🔄 結果を確定して午後の予測を最適化"):
                    st.rerun()

        with tab2:
            st.write("### 今日の配置的中バイアス")
            hits = full_df[full_df['着順'] <= 3]
            if not hits.empty:
                st.plotly_chart(px.bar(hits['タイプ'].value_counts(), title="的中パターンの分布"))
                st.write(f"現在の青塗ボーナス値: {full_df['当日バイアス'].max():.1f}")
            else:
                st.info("結果が入力されると、今日の流れ（バイアス）が表示されます。")
