import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import pickle
import os

# --- 1. 基本設定 ---
st.set_page_config(page_title="配置馬券術 分析システム", layout="wide")

# AIモデル読み込み
@st.cache_resource
def load_ai_model():
    MODEL_PATH = 'model.pkl'
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except: return None
    return None

model = load_ai_model()

# 重複列名の回避
def make_cols_unique(df):
    cols = []
    counts = {}
    for col in df.columns:
        c_str = str(col).strip() if pd.notna(col) else "Unnamed"
        if c_str in counts:
            counts[c_str] += 1
            cols.append(f"{c_str}_{counts[c_str]}")
        else:
            counts[c_str] = 0
            cols.append(c_str)
    df.columns = cols
    return df

def to_half_width(text):
    if pd.isna(text): return text
    text = str(text)
    table = str.maketrans('０１２３４５６７８９．', '0123456789.')
    return re.sub(r'[^\d\.]', '', str(text).translate(table))

def normalize_name(x):
    if pd.isna(x): return ''
    s = str(x).strip().replace('　', '').replace(' ', '')
    s = re.split(r'[,(（/]', s)[0]
    return re.sub(r'[★☆▲△◇$*]', '', s)

# --- 2. データ読み込み（自動マッピング強化） ---
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.xlsx'):
            df_raw = pd.read_excel(file, header=None, engine='openpyxl')
        else:
            try: df_raw = pd.read_csv(file, header=None, encoding='utf-8')
            except: df_raw = pd.read_csv(file, header=None, encoding='cp932')
        
        # 見出し行を自動探索
        best_row, max_hits = 0, 0
        keywords = ['場所', '場名', 'R', 'レース', '番', '馬名', 'オッズ', '単勝']
        for i in range(min(len(df_raw), 25)):
            row_str = "".join(df_raw.iloc[i].astype(str))
            hits = sum(1 for k in keywords if k in row_str)
            if hits > max_hits:
                max_hits, best_row = hits, i
        
        df = df_raw.iloc[best_row:].reset_index(drop=True)
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
        df = make_cols_unique(df)

        # 列名の自動マッピング
        mapping_rules = {
            'R': ['R', 'Ｒ', 'レース', '番組'],
            '場名': ['場所', '場名', '競馬場', '会場', '開催'],
            '正番': ['番', '馬番', '正番', '枠番'],
            '馬名': ['馬名', '馬', '名称'],
            '単ｵｯｽﾞ': ['オッズ', '単勝', '単ｵｯｽﾞ', '単オッズ'],
            '騎手': ['騎手', 'ジョッキー'],
            '厩舎': ['厩舎', '調教師'],
            '馬主': ['馬主', 'オーナー'],
            '着順': ['着順', '着', '順位']
        }
        
        col_map = {}
        for internal_name, keys in mapping_rules.items():
            for c in df.columns:
                if any(k in str(c) for k in keys):
                    col_map[c] = internal_name; break
        df = df.rename(columns=col_map)
        df = make_cols_unique(df)

        # 必須列の型変換
        if 'R' in df.columns: df['R'] = pd.to_numeric(df['R'].apply(to_half_width).astype(str).str.extract(r'(\d+)', expand=False), errors='coerce').fillna(0).astype(int)
        if '正番' in df.columns: df['正番'] = pd.to_numeric(df['正番'].apply(to_half_width).astype(str).str.extract(r'(\d+)', expand=False), errors='coerce').fillna(0).astype(int)
        if '単ｵｯｽﾞ' in df.columns: df['単ｵｯｽﾞ'] = pd.to_numeric(df['単ｵｯｽﾞ'].apply(to_half_width).astype(str).str.extract(r'(\d+\.?\d*)', expand=False), errors='coerce').fillna(99.0)
        
        for col in ['場名', '馬名', '騎手', '厩舎', '馬主']:
            if col in df.columns: df[col] = df[col].astype(str).apply(normalize_name)
            else: df[col] = ""

        return df[df['R'] > 0].copy(), "success"
    except Exception as e: return pd.DataFrame(), str(e)

# --- 3. 解析・AIエンジン ---
def analyze_haichi(df_curr, df_prev=None):
    df = df_curr.copy()
    if 'タイプ' in df.columns and df['タイプ'].notna().any(): return df

    df['青塗フラグ'] = 0; df['ペアフラグ'] = 0; df['前日配置フラグ'] = 0; df['スコア'] = 0.0
    df['頭数'] = df.groupby(['場名', 'R'])['正番'].transform('max').fillna(16).astype(int)
    df['逆番'] = (df['頭数'] + 1) - df['正番']
    df['正循環'] = df['頭数'] + df['正番']
    df['逆循環'] = df['頭数'] + df['逆番']
    df['タイプ_list'] = [[] for _ in range(len(df))]
    idx_map = {(row['場名'], row['R'], row['正番']): idx for idx, row in df.iterrows()}

    # 青塗判定
    for col in ['騎手', '厩舎', '馬主']:
        if col not in df.columns: continue
        g_keys = ['場名', col] if col == '騎手' else [col]
        for name, group in df.groupby(g_keys):
            if len(group) < 2 or not str(name).strip(): continue
            all_sets = [{r['正番'], r['逆番'], r['正循環'], r['逆循環']} for _, r in group.iterrows()]
            common = set.intersection(*all_sets)
            if common:
                for _, row in group.iterrows():
                    idx = idx_map.get((row['場名'], row['R'], row['正番']))
                    if idx is not None:
                        df.at[idx, '青塗フラグ'] = 1; df.at[idx, 'タイプ_list'].append(f'★{col}青塗'); df.at[idx, 'スコア'] += 9.2

    # AI予測
    if model:
        try:
            X = df[['単ｵｯｽﾞ', '青塗フラグ', 'ペアフラグ', '前日配置フラグ']].fillna(0)
            probs = model.predict_proba(X)
            df['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
        except: df['AI激走確率'] = 0.0
    else: df['AI激走確率'] = 0.0

    df['タイプ'] = df['タイプ_list'].apply(lambda x: ' / '.join(x) if x else '無')
    return df

def apply_ranking_logic(df_in):
    df = df_in.copy()
    df['総合スコア'] = df['スコア'] + (df.get('AI激走確率', 0) / 10.0)
    df['評価'] = df['総合スコア'].apply(lambda x: "👑軸" if x>=15 else "🔥注" if x>=10 else "▲")
    return df

# --- 4. UI ---
st.title("🏇 配置馬券 AI分析・管理システム")

with st.sidebar:
    st.header("📂 読み込み")
    up_curr = st.file_uploader("当日データ", type=['xlsx', 'csv'], key="curr")
    if up_curr and 'analyzed_df' in st.session_state:
        st.divider(); st.header("💾 保存")
        csv = st.session_state['analyzed_df'].to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 分析CSVを保存", csv, f"progress_{up_curr.name}.csv")

if up_curr:
    df_raw, status = load_data(up_curr)
    if status == "success":
        if 'analyzed_df' not in st.session_state:
            st.session_state['analyzed_df'] = apply_ranking_logic(analyze_haichi(df_raw))
        
        full_df = st.session_state['analyzed_df']

        # ① 結果入力フォーム
        st.subheader("📝 予測・結果入力")
        with st.form("result_form"):
            places = sorted(full_df['場名'].unique())
            p_tabs = st.tabs(places); edited_dfs = []
            for p_tab, place in zip(p_tabs, places):
                with p_tab:
                    p_df = full_df[full_df['場名'] == place]
                    r_tabs = st.tabs([f"{r}R" for r in sorted(p_df['R'].unique())])
                    for r_tab, r_num in zip(r_tabs, sorted(p_df['R'].unique())):
                        with r_tab:
                            race_full = p_df[p_df['R'] == r_num].sort_values('正番')
                            # 【KeyError対策】存在する列だけを表示対象にする
                            potential_cols = ['評価','正番','馬名','単ｵｯｽﾞ','AI激走確率','タイプ','総合スコア','着順']
                            disp_cols = [c for c in potential_cols if c in race_full.columns]
                            
                            ed = st.data_editor(race_full[disp_cols], hide_index=True, use_container_width=True, key=f"ed_{place}_{r_num}",
                                                column_config={"AI激走確率": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})
                            
                            # 更新用データの保持
                            updated = race_full.copy()
                            for _, row in ed.iterrows():
                                if '着順' in row: updated.loc[updated['正番'] == row['正番'], '着順'] = row['着順']
                            edited_dfs.append(updated)
            
            # 【Missing Submit Button対策】フォームの最後にボタンを配置
            submit = st.form_submit_button("🔄 入力を確定して全データを更新")
            if submit:
                st.session_state['analyzed_df'] = apply_ranking_logic(pd.concat(edited_dfs, ignore_index=True))
                st.rerun()

        # ② 統計表示
        st.divider(); st.subheader("📈 的中統計")
        df_res = full_df[full_df['着順'].notna()].copy()
        if not df_res.empty:
            df_fk = df_res[df_res['着順'] <= 3]
            c1, c2 = st.columns([1, 2])
            with c1: st.metric("複勝率", f"{len(df_fk)/len(df_res)*100:.1f}%"); st.metric("的中数", len(df_fk))
            with c2:
                all_p = [p for pats in df_fk['タイプ'] for p in str(pats).split(' / ') if '無' not in p]
                if all_p: st.plotly_chart(px.pie(pd.Series(all_p).value_counts().reset_index(), values='count', names='index', hole=0.4), use_container_width=True)
else:
    st.info("サイドバーからファイルを読み込んでください。")
