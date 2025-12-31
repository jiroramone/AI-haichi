import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os

# --- 1. 基本設定 ---
st.set_page_config(page_title="全自動・配置予測システム", layout="wide")

# AIモデルの読み込み
@st.cache_resource
def get_model():
    if os.path.exists('model.pkl'):
        try:
            with open('model.pkl', 'rb') as f:
                return pickle.load(f)
        except: return None
    return None

model = get_model()

# --- 2. 自動クリーニング関数群 ---
def make_unique(cols):
    """重複した列名を自動リネーム"""
    new_cols = []
    counts = {}
    for c in cols:
        c_str = str(c).strip() if pd.notna(c) else "Unnamed"
        if c_str in counts:
            counts[c_str] += 1
            new_cols.append(f"{c_str}_{counts[c_str]}")
        else:
            counts[c_str] = 0
            new_cols.append(c_str)
    return new_cols

def auto_extract_num(val):
    """'1R'や'15.5倍'から数字だけを抽出"""
    if pd.isna(val): return 0
    match = re.search(r'(\d+\.?\d*)', str(val).replace(',', ''))
    return float(match.group(1)) if match else 0

# --- 3. 自動読み込みエンジン ---
def auto_load_and_analyze(file):
    try:
        if file.name.endswith('.xlsx'):
            df_raw = pd.read_excel(file, header=None)
        else:
            try: df_raw = pd.read_csv(file, header=None, encoding='utf-8')
            except: df_raw = pd.read_csv(file, header=None, encoding='cp932')
        
        # 【自動探索】本物のヘッダー行（項目名がある行）を探す
        best_row = 0
        max_hits = 0
        keywords = ['場所', '場名', 'R', 'レース', '馬番', '番', '馬名', 'オッズ', '単勝']
        
        for i in range(min(len(df_raw), 30)):
            row_str = "".join(df_raw.iloc[i].astype(str))
            hits = sum(1 for k in keywords if k in row_str)
            if hits > max_hits:
                max_hits = hits
                best_row = i
        
        # ヘッダー設定と重複排除
        df = df_raw.iloc[best_row:].reset_index(drop=True)
        df.columns = make_unique(df.iloc[0])
        df = df.iloc[1:].reset_index(drop=True)

        # 【自動名寄せ】列名のマッチング
        col_map = {}
        mapping_rules = {
            'R': ['R', 'Ｒ', 'レース', '番組'],
            '場名': ['場所', '場名', '競馬場', '会場', '開催'],
            '正番': ['番', '馬番', '正番', '枠番'],
            '馬名': ['馬名', '馬', '名称'],
            '単ｵｯｽﾞ': ['オッズ', '単勝', '単ｵｯｽﾞ', '単オッズ'],
            '騎手': ['騎手', 'ジョッキー'],
            '厩舎': ['厩舎', '調教師']
        }

        for internal_name, keys in mapping_rules.items():
            for c in df.columns:
                if any(k in str(c) for k in keys):
                    col_map[c] = internal_name
                    break
        
        df = df.rename(columns=col_map)
        
        # 必須列の補完（無い場合は0埋め）
        for c in ['R', '場名', '正番', '馬名', '単ｵｯｽﾞ']:
            if c not in df.columns: df[c] = "" if c in ['場名', '馬名'] else 0

        # 型の自動変換
        df['R'] = df['R'].apply(auto_extract_num).astype(int)
        df['正番'] = df['正番'].apply(auto_extract_num).astype(int)
        df['単ｵｯｽﾞ'] = df['単ｵｯｽﾞ'].apply(auto_extract_num)
        df['場名'] = df['場名'].astype(str).str.strip().replace('　', '')

        # 1R以上の有効行のみ
        df = df[df['R'] > 0].copy()

        # 配置解析（青塗フラグ）
        df['青塗フラグ'] = 0
        for col in ['騎手', '厩舎']:
            if col in df.columns:
                mask = (df[col].astype(str).str.strip() != "")
                dup = df[mask].duplicated(subset=['場名', col], keep=False)
                df.loc[df.index[mask][dup], '青塗フラグ'] = 1

        # AI予測
        if model and not df.empty:
            try:
                X = pd.DataFrame({
                    '単ｵｯｽﾞ': df['単ｵｯｽﾞ'].replace(0, 99.0),
                    '青塗フラグ': df['青塗フラグ'],
                    'ペアフラグ': 0, '前日配置フラグ': 0
                }).fillna(0)
                probs = model.predict_proba(X)
                df['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
            except: df['AI激走確率'] = 0.0
        else:
            df['AI激走確率'] = 0.0

        return df
    except Exception as e:
        st.error(f"自動読み取りに失敗しました: {e}")
        return pd.DataFrame()

# --- 4. メイン UI ---
st.title("🏇 AI配置予測（全自動・高速読込版）")

file = st.sidebar.file_uploader("当日配置表をアップロード", type=['xlsx', 'csv'])

if file:
    # 初回読み込み
    if 'auto_df' not in st.session_state or st.sidebar.button("🔄 データを再読込"):
        with st.spinner('データを自動スキャン中...'):
            st.session_state['auto_df'] = auto_load_and_analyze(file)

    df = st.session_state['auto_df']

    if not df.empty:
        # 会場とレースの選択
        places = sorted([p for p in df['場名'].unique() if p != ""])
        if places:
            st.sidebar.divider()
            target_place = st.sidebar.selectbox("会場を選択", places)
            r_list = sorted(df[df['場名'] == target_place]['R'].unique())
            target_r = st.sidebar.selectbox("レースを選択", r_list)

            # 表示
            view = df[(df['場名'] == target_place) & (df['R'] == target_r)].sort_values('AI激走確率', ascending=False)
            
            st.subheader(f"📊 {target_place} {target_r}R 予測結果")
            
            # 推奨馬をカード表示
            top = view.iloc[0]
            st.info(f"👑 AI推奨馬: {top['正番']}番 {top['馬名']} (確率 {top['AI激走確率']}%)")

            st.dataframe(
                view[['正番', '馬名', '単ｵｯｽﾞ', 'AI激走確率', '青塗フラグ']],
                column_config={
                    "AI激走確率": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                    "単ｵｯｽﾞ": st.column_config.NumberColumn(format="%.1f")
                },
                use_container_width=True, hide_index=True
            )
        else:
            st.warning("会場名を自動特定できませんでした。エクセルの内容を確認してください。")
    else:
        st.error("有効なデータが見つかりませんでした。")
else:
    st.info("👈 サイドバーからファイルをアップロードしてください。すべて自動で解析します。")
