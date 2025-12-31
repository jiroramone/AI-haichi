import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os

# --- 1. 基本設定 ---
st.set_page_config(page_title="配置馬券 AI激走予測システム", layout="wide")

# AIモデルの読み込み
MODEL_PATH = 'model.pkl'
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

# --- 2. 便利関数群 ---
def to_half_width(text):
    if pd.isna(text): return text
    text = str(text)
    table = str.maketrans('０１２３４５６７８９．', '0123456789.')
    return re.sub(r'[^\d\.]', '', text.translate(table))

def normalize_name(x):
    if pd.isna(x): return ''
    s = str(x).strip().replace('　', '').replace(' ', '')
    s = re.split(r'[,(（/]', s)[0]
    return re.sub(r'[★☆▲△◇$*]', '', s)

# --- 3. 配置解析エンジン (スコア計算) ---
# ※学習時と同じロジックで「総合スコア」を算出する必要があります
def run_haichi_engine(df_curr, df_prev=None):
    df = df_curr.copy()
    
    # 基本情報の整理
    max_umaban = df.groupby(['場名', 'R'])['正番'].transform('max')
    df['頭数'] = max_umaban.fillna(16).astype(int)
    df['逆番'] = (df['頭数'] + 1) - df['正番']
    df['正循環'] = df['頭数'] + df['正番']
    df['逆循環'] = df['頭数'] + df['逆番']
    
    df['スコア'] = 0.0
    idx_map = {(row['場名'], row['R'], row['正番']): idx for idx, row in df.iterrows()}
    
    # A. 青塗判定 (馬主・厩舎・騎手の重なり)
    for col in ['騎手', '厩舎', '馬主']:
        g_keys = ['場名', col] if col == '騎手' else [col]
        for name, group in df.groupby(g_keys):
            if len(group) < 2 or not name: continue
            all_sets = [{r['正番'], r['逆番'], r['正循環'], r['逆循環']} for _, r in group.iterrows()]
            common = set.intersection(*all_sets)
            if common:
                for _, row in group.iterrows():
                    idx = idx_map.get((row['場名'], row['R'], row['正番']))
                    if idx is not None:
                        df.at[idx, 'スコア'] += 9.2 # 青塗スコア
    
    # B. ペア判定 (A-Rパターン)
    pair_labels = list("ABCDEFGHIJKLMNOP")
    for col in ['騎手', '厩舎', '馬主']:
        for name, group in df.groupby(['場名', col] if col=='騎手' else col):
            if len(group) < 2 or not name: continue
            rows = group.sort_values('R').to_dict('records')
            for i in range(len(rows)-1):
                r1, r2 = rows[i], rows[i+1]
                v1 = [r1[c] for c in ['正番','逆番','正循環','逆循環']]
                v2 = [r2[c] for c in ['正番','逆番','正循環','逆循環']]
                pats = [pair_labels[x*4+y] for x in range(4) for y in range(4) if v1[x]==v2[y] and v1[x]!=0]
                if pats:
                    is_c = any(p in pats for p in ['C','D','G','H'])
                    for r_data in [r1, r2]:
                        idx = idx_map.get((r_data['場名'], r_data['R'], r_data['正番']))
                        if idx is not None:
                            df.at[idx, 'スコア'] += 4.0 if is_c else 3.0

    # C. 前日配置判定
    if df_prev is not None and not df_prev.empty:
        for idx, row in df.iterrows():
            prev_match = df_prev[(df_prev['場名'] == row['場名']) & (df_prev['R'] == row['R']) & (df_prev['騎手'] == row['騎手'])]
            for _, p_row in prev_match.iterrows():
                if {row['正番'],row['逆番'],row['正循環'],row['逆循環']}.intersection({p_row['正番'],p_row['逆番'],p_row['正循環'],p_row['逆循環']}):
                    df.at[idx, 'スコア'] += 8.3

    df['総合スコア'] = df['スコア']
    return df

# --- 4. データ読み込み ---
def load_data(file):
    try:
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        else:
            try: df = pd.read_csv(file, encoding='utf-8')
            except: df = pd.read_csv(file, encoding='cp932')
        
        # ヘッダー位置自動調整
        for i in range(min(len(df), 20)):
            row_vals = [str(x) for x in df.iloc[i].values]
            if any('場所' in x or 'R' in x or '馬名' in x for x in row_vals):
                df.columns = df.iloc[i]
                df = df.iloc[i+1:].reset_index(drop=True)
                break
        
        df.columns = df.columns.astype(str).str.strip()
        name_map = {'場所':'場名','競馬場':'場名','開催':'場名','レース':'R','Ｒ':'R','番':'正番','馬番':'正番','単勝オッズ':'単ｵｯｽﾞ','オッズ':'単ｵｯｽﾞ'}
        df = df.rename(columns=name_map)
        
        # 数値変換とクリーンアップ
        df['R'] = pd.to_numeric(df['R'].apply(to_half_width), errors='coerce')
        df['正番'] = pd.to_numeric(df['正番'].apply(to_half_width), errors='coerce')
        df['単ｵｯｽﾞ'] = pd.to_numeric(df['単ｵｯｽﾞ'].apply(to_half_width), errors='coerce')
        df = df.dropna(subset=['R', '正番'])
        
        for col in ['騎手', '厩舎', '馬主', '馬名', '場名']:
            if col in df.columns: df[col] = df[col].astype(str).apply(normalize_name)
            else: df[col] = ""
            
        return df
    except Exception as e:
        st.error(f"読み込みエラー: {e}")
        return pd.DataFrame()

# --- 5. UI と メイン処理 ---
st.title("🏇 配置馬券 AI激走予測アプリ")


with st.sidebar:
    st.header("📂 予想データの読み込み")
    up_curr = st.file_uploader("今日の出馬表(Excel/CSV)", type=['xlsx', 'csv'])
    up_prev = st.file_uploader("前日の配置表（任意）", type=['xlsx', 'csv'])
    st.divider()
    if model:
        st.success("🤖 AIモデル: 読込済")
    else:
        st.error("⚠️ AIモデル(model.pkl)がありません")

if up_curr:
    df_raw = load_data(up_curr)
    df_p_raw = load_data(up_prev) if up_prev else None
    
    if not df_raw.empty:
        # 配置スコア計算
        df_analyzed = run_haichi_engine(df_raw, df_p_raw)
        
        # AI予想実行
        if model:
            X = df_analyzed[['正番', '単ｵｯｽﾞ', '総合スコア']].copy()
            X['単ｵｯｽﾞ'] = X['単ｵｯｽﾞ'].fillna(99.0)
            X['総合スコア'] = X['総合スコア'].fillna(0.0)
            
            # 3着内確率を算出
            probs = model.predict_proba(X)
            df_analyzed['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
        
        # 会場ごとに表示
        places = sorted(df_analyzed['場名'].unique())
        tabs = st.tabs(places)
        
        for tab, place in zip(tabs, places):
            with tab:
                p_df = df_analyzed[df_analyzed['場名'] == place]
                r_nums = sorted(p_df['R'].unique().astype(int))
                r_num = st.selectbox(f"レース選択", r_nums, key=f"sel_{place}")
                
                res = p_df[p_df['R'] == r_num].sort_values('AI激走確率', ascending=False)
                
                st.subheader(f"📊 {place} {r_num}R AI予測ランキング")
                
                # 表示列の整理
                disp_cols = ['正番', '馬名', '単ｵｯｽﾞ', '総合スコア', 'AI激走確率']
                
                # スタイル設定
                def color_prob(val):
                    color = 'red' if val >= 35 else 'orange' if val >= 20 else 'black'
                    return f'color: {color}; font-weight: bold'

                st.dataframe(
                    res[disp_cols].style.applymap(color_prob, subset=['AI激走確率']),
                    use_container_width=True, hide_index=True
                )
                
                st.info("💡 確率が高い順に表示しています。35%以上は激アツです。")
else:
    st.info("サイドバーから当日のファイルをアップロードしてください。")
