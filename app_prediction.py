import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os
import time

# --- 1. 基本設定 ---
st.set_page_config(page_title="配置馬券 AI激走予測システム", layout="wide")

# AIモデルの読み込み
MODEL_PATH = 'model.pkl'
model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except:
        st.error("model.pkl の読み込みに失敗しました。ファイルが壊れているか形式が違います。")

# 重複カラム回避
def fix_duplicate_cols(df):
    cols = []
    counts = {}
    for col in df.columns:
        c_name = str(col).strip() if pd.notna(col) else "Unnamed"
        if c_name in counts:
            counts[c_name] += 1
            cols.append(f"{c_name}_{counts[c_name]}")
        else:
            counts[c_name] = 0
            cols.append(c_name)
    df.columns = cols
    return df

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

# --- 2. データ読み込み（「R」エラー対策版） ---
def load_data(file):
    try:
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        else:
            try: df = pd.read_csv(file, encoding='utf-8')
            except: df = pd.read_csv(file, encoding='cp932')
        
        # 1. 読み込み直後の重複回避
        df = fix_duplicate_cols(df)

        # 2. 項目名（ヘッダー）の探索
        header_found = False
        for i in range(min(len(df), 20)):
            row_vals = [str(x) for x in df.iloc[i].values]
            # 「R」や「場所」が含まれる行を探す
            if any(re.search(r'^[RrＲｒ]$|^場所$|^馬名$', str(x).strip()) for x in row_vals):
                df.columns = df.iloc[i]
                df = df.iloc[i+1:].reset_index(drop=True)
                df = fix_duplicate_cols(df)
                header_found = True
                break
        
        # 3. 列名の名寄せ（より強力な部分一致）
        new_cols = []
        for c in df.columns:
            c_str = str(c).strip()
            if re.search(r'^[RrＲｒ]$|レース', c_str): new_cols.append('R')
            elif re.search(r'場所|競馬場|開催', c_str): new_cols.append('場名')
            elif re.search(r'^番$|馬番|正番', c_str): new_cols.append('正番')
            elif re.search(r'単勝オッズ|単ｵｯｽﾞ|オッズ', c_str): new_cols.append('単ｵｯｽﾞ')
            else: new_cols.append(c_str)
        df.columns = new_cols
        df = fix_duplicate_cols(df)

        # 4. 必須列の存在確認
        missing = [c for c in ['R', '場名', '正番'] if c not in df.columns]
        if missing:
            st.error(f"必須項目 {missing} が見つかりません。現在の列名: {list(df.columns)}")
            return pd.DataFrame()

        # 5. 型変換
        df['R'] = pd.to_numeric(df['R'].apply(to_half_width), errors='coerce')
        df['正番'] = pd.to_numeric(df['正番'].apply(to_half_width), errors='coerce')
        df['単ｵｯｽﾞ'] = pd.to_numeric(df['単ｵｯｽﾞ'].apply(to_half_width), errors='coerce')
        
        df = df.dropna(subset=['R', '正番'])
        df['R'] = df['R'].astype(int)
        df['正番'] = df['正番'].astype(int)
        
        for col in ['騎手', '厩舎', '馬主', '馬名', '場名']:
            if col in df.columns: df[col] = df[col].astype(str).apply(normalize_name)
            else: df[col] = ""
            
        return df
    except Exception as e:
        st.error(f"読み込み処理中に重大なエラーが発生しました: {e}")
        return pd.DataFrame()

# --- 3. 配置解析エンジン (スコア計算) ---
def run_haichi_engine(df_curr, df_prev=None):
    df = df_curr.copy()
    
    # 基本情報の整理
    df['頭数'] = df.groupby(['場名', 'R'])['正番'].transform('max').fillna(16).astype(int)
    df['逆番'] = (df['頭数'] + 1) - df['正番']
    df['正循環'] = df['頭数'] + df['正番']
    df['逆循環'] = df['頭数'] + df['逆番']
    
    df['総合スコア'] = 0.0
    idx_map = {(row['場名'], row['R'], row['正番']): idx for idx, row in df.iterrows()}
    
    # A. 青塗判定
    for col in ['騎手', '厩舎', '馬主']:
        g_keys = ['場名', col] if col == '騎手' else [col]
        for name, group in df.groupby(g_keys):
            if len(group) < 2 or not str(name).strip(): continue
            all_sets = [{r['正番'], r['逆番'], r['正循環'], r['逆循環']} for _, r in group.iterrows()]
            common = set.intersection(*all_sets)
            if common:
                for _, row in group.iterrows():
                    idx = idx_map.get((row['場名'], row['R'], row['正番']))
                    if idx is not None: df.at[idx, '総合スコア'] += 9.2
    
    # B. ペア判定
    pair_labels = list("ABCDEFGHIJKLMNOP")
    for col in ['騎手', '厩舎', '馬主']:
        for name, group in df.groupby(['場名', col] if col=='騎手' else col):
            if len(group) < 2 or not str(name).strip(): continue
            rows = group.sort_values('R').to_dict('records')
            for i in range(len(rows)-1):
                r1, r2 = rows[i], rows[i+1]
                v1 = [r1[c] for c in ['正番','逆番','正循環','逆循環']]
                v2 = [r2[c] for c in ['正番','逆番','正循環','逆循環']]
                pats = [pair_labels[x*4+y] for x in range(4) for y in range(4) if v1[x]==v2[y]]
                if pats:
                    score = 4.0 if any(p in pats for p in ['C','D','G','H']) else 3.0
                    for r_data in [r1, r2]:
                        idx = idx_map.get((r_data['場名'], r_data['R'], r_data['正番']))
                        if idx is not None: df.at[idx, '総合スコア'] += score

    # C. 前日配置
    if df_prev is not None and not df_prev.empty:
        for idx, row in df.iterrows():
            prev_match = df_prev[(df_prev['場名'] == row['場名']) & (df_prev['R'] == row['R']) & (df_prev['騎手'] == row['騎手'])]
            for _, p_row in prev_match.iterrows():
                if {row['正番'],row['逆番'],row['正循環'],row['逆循環']}.intersection({p_row['正番'],p_row['逆番'],p_row['正循環'],p_row['逆循環']}):
                    df.at[idx, '総合スコア'] += 8.3

    return df

# --- 4. UI 画面 ---
st.title("🏇 配置馬券 AI激走予測アプリ")

with st.sidebar:
    st.header("📂 データ読み込み")
    up_curr = st.file_uploader("今日の出馬表(Excel/CSV)", type=['xlsx', 'csv'])
    up_prev = st.file_uploader("前日の配置表（任意）", type=['xlsx', 'csv'])
    st.divider()
    if model: st.success("🤖 AIモデル: 読込済")
    else: st.error("⚠️ model.pkl が見つかりません")

if up_curr:
    df_raw = load_data(up_curr)
    df_p_raw = load_data(up_prev) if up_prev else None
    
    if not df_raw.empty:
        # スコア計算
        df_analyzed = run_haichi_engine(df_raw, df_p_raw)
        
        # AI予測
        if model:
            features = ['正番', '単ｵｯｽﾞ', '総合スコア']
            X = df_analyzed[features].copy()
            X['単ｵｯｽﾞ'] = X['単ｵｯｽﾞ'].fillna(99.0)
            X['総合スコア'] = X['総合スコア'].fillna(0.0)
            
            probs = model.predict_proba(X)
            df_analyzed['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
        
        # 会場表示
        places = sorted(df_analyzed['場名'].unique())
        tabs = st.tabs(places)
        
        for tab, place in zip(tabs, places):
            with tab:
                p_df = df_analyzed[df_analyzed['場名'] == place]
                r_num = st.selectbox(f"レース選択 ({place})", sorted(p_df['R'].unique().astype(int)), key=f"sel_{place}")
                
                res = p_df[p_df['R'] == r_num].sort_values('AI激走確率' if model else '正番', ascending=False if model else True)
                
                st.subheader(f"📊 {place} {r_num}R 予測結果")
                disp_cols = ['正番', '馬名', '単ｵｯｽﾞ', '総合スコア']
                if 'AI激走確率' in res.columns: disp_cols.append('AI激走確率')

                def color_prob(val):
                    c = 'red' if val >= 35 else 'orange' if val >= 20 else 'black'
                    return f'color: {c}; font-weight: bold'

                st.dataframe(
                    res[disp_cols].style.applymap(color_prob, subset=['AI激走確率']) if model else res[disp_cols],
                    use_container_width=True, hide_index=True
                )
