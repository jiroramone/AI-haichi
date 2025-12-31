import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os

# --- 1. 基本設定 ---
st.set_page_config(page_title="AI配置馬券 予測システム", layout="wide")

# AIモデル読込（エラー時はNoneにする）
MODEL_PATH = 'model.pkl'
model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        st.sidebar.error(f"モデル読込失敗: {e}")

def to_half_width(text):
    if pd.isna(text): return text
    table = str.maketrans('０１２３４５６７８９．', '0123456789.')
    return re.sub(r'[^\d\.]', '', str(text).translate(table))

def normalize_name(x):
    if pd.isna(x): return ''
    s = str(x).strip().replace('　', '').replace(' ', '')
    return re.split(r'[,(（/]', s)[0]

# --- 2. データ読み込み（認識能力を最大化） ---
def load_data(file):
    try:
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        else:
            try: df = pd.read_csv(file, encoding='utf-8')
            except: df = pd.read_csv(file, encoding='cp932')
        
        # 項目名を探す（20行目までスキャン）
        for i in range(min(len(df), 20)):
            row_vals = [str(x) for x in df.iloc[i].values]
            if any(re.search(r'^[RrＲｒ]$|場所|馬名|番', str(x).strip()) for x in row_vals):
                df.columns = df.iloc[i]; df = df.iloc[i+1:].reset_index(drop=True); break

        df.columns = [str(c).strip() for c in df.columns]
        
        # 項目名の名寄せ（R, 場名, 正番, 単ｵｯｽﾞ）
        name_map = {
            '場所':'場名','開催':'場名','競馬場':'場名',
            'レース':'R','Ｒ':'R','ｒ':'R',
            '番':'正番','馬番':'正番','正番':'正番',
            '単勝オッズ':'単ｵｯｽﾞ','オッズ':'単ｵｯｽﾞ','単ｵｯズ':'単ｵｯｽﾞ'
        }
        # 部分一致でも置換
        for col in df.columns:
            for k, v in name_map.items():
                if k in col:
                    df = df.rename(columns={col: v})
                    break
        
        # 必須項目のクリーニング
        if 'R' in df.columns and '正番' in df.columns:
            df['R'] = pd.to_numeric(df['R'].apply(to_half_width), errors='coerce')
            df['正番'] = pd.to_numeric(df['正番'].apply(to_half_width), errors='coerce')
            df = df.dropna(subset=['R', '正番'])
            df['R'] = df['R'].astype(int); df['正番'] = df['正番'].astype(int)
            
            if '単ｵｯｽﾞ' in df.columns:
                df['単ｵｯｽﾞ'] = pd.to_numeric(df['単ｵｯｽﾞ'].apply(to_half_width), errors='coerce').fillna(99.0)
            else:
                df['単ｵｯｽﾞ'] = 99.0

            for col in ['騎手', '厩舎', '馬主', '馬名', '場名']:
                if col in df.columns: df[col] = df[col].astype(str).apply(normalize_name)
                else: df[col] = ""
            return df.copy()
        else:
            st.error("ファイル内に「R」または「番」の列が見つかりません。")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"ファイル読込エラー: {e}")
        return pd.DataFrame()

# --- 3. 解析・AI予測（エラーでも表を出す設計） ---
def run_full_analysis(df):
    try:
        # 初期化
        df['青塗フラグ'] = 0; df['ペアフラグ'] = 0; df['前日配置フラグ'] = 0; df['AI激走確率'] = 0.0; df['タイプ'] = ""

        # 配置計算の準備
        df['頭数'] = df.groupby(['場名', 'R'])['正番'].transform('max').fillna(16).astype(int)
        df['逆番'] = (df['頭数'] + 1) - df['正番']
        df['正循環'] = df['頭数'] + df['正番']
        df['逆循環'] = df['頭数'] + df['逆番']
        idx_map = {(row['場名'], row['R'], row['正番']): idx for idx, row in df.iterrows()}

        # A. 青塗判定（簡略化）
        for col in ['騎手', '厩舎', '馬主']:
            if col not in df.columns: continue
            g_keys = ['場名', col] if col == '騎手' else [col]
            for _, group in df.groupby(g_keys):
                if len(group) < 2: continue
                all_sets = [{r['正番'], r['逆番'], r['正循環'], r['逆循環']} for _, r in group.iterrows()]
                common = set.intersection(*all_sets)
                if common:
                    for _, row in group.iterrows():
                        idx = idx_map.get((row['場名'], row['R'], row['正番']))
                        if idx is not None:
                            df.at[idx, '青塗フラグ'] = 1; df.at[idx, 'タイプ'] += f"★{col}青塗 "

        # B. AI予測実行（エラーハンドリング付き）
        if model is not None:
            try:
                # 特徴量リスト：学習時と完全に一致させる
                features = ['単ｵｯｽﾞ', '青塗フラグ', 'ペアフラグ', '前日配置フラグ']
                # 足りない列を強制作成
                for f in features:
                    if f not in df.columns: df[f] = 0
                
                X = df[features].fillna(0)
                probs = model.predict_proba(X)
                df['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
            except Exception as e:
                st.sidebar.warning(f"AI予測中にエラーが発生しましたが、解析を続行します。")
        
        return df
    except Exception as e:
        st.error(f"解析エラー: {e}")
        return df

# --- 4. UI 表示 ---
st.title("🏇 AI配置予測・全頭表示システム")

up_file = st.sidebar.file_uploader("当日配置表(Excel/CSV)", type=['xlsx', 'csv'])

if up_file:
    # ファイルがアップロードされたら解析（初回のみ）
    if 'full_df' not in st.session_state or st.sidebar.button("🔄 データを再解析"):
        df_raw = load_data(up_file)
        if not df_raw.empty:
            st.session_state['full_df'] = run_full_analysis(df_raw)
        else:
            st.stop()

    df = st.session_state['full_df']
    
    # 選択メニュー
    places = sorted(df['場名'].unique())
    if places:
        place = st.sidebar.selectbox("会場選択", places)
        r_nums = sorted(df[df['場名'] == place]['R'].unique())
        r_num = st.sidebar.selectbox("レース選択", r_nums)

        # 抽出
        res = df[(df['場名'] == place) & (df['R'] == r_num)].sort_values('AI激走確率', ascending=False)

        # 表示
        st.subheader(f"📊 {place} {r_num}R 予測結果")
        st.dataframe(
            res[['正番', '馬名', '単ｵｯｽﾞ', 'AI激走確率', 'タイプ', '着順']],
            column_config={
                "AI激走確率": st.column_config.ProgressColumn("激走確率", format="%.1f%%", min_value=0, max_value=100),
                "単ｵｯｽﾞ": st.column_config.NumberColumn("オッズ", format="%.1f")
            },
            use_container_width=True, hide_index=True
        )
    else:
        st.warning("表示できる会場データがありません。")

else:
    st.info("左側のサイドバーからファイルをアップロードしてください。")
