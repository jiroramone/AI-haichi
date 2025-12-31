import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os

# --- 1. 基本設定 ---
st.set_page_config(page_title="配置予測・重複エラー対策版", layout="wide")

# カラム名の重複を解消する関数
def make_columns_unique(cols):
    new_cols = []
    counts = {}
    for col in cols:
        col_str = str(col).strip() if pd.notna(col) else "Unnamed"
        if col_str in counts:
            counts[col_str] += 1
            new_cols.append(f"{col_str}_{counts[col_str]}")
        else:
            counts[col_str] = 0
            new_cols.append(col_str)
    return new_cols

@st.cache_resource
def load_ai_model():
    if os.path.exists('model.pkl'):
        try:
            with open('model.pkl', 'rb') as f:
                return pickle.load(f)
        except: return None
    return None

model = load_ai_model()

def clean_numeric(val):
    if pd.isna(val): return 0
    s = str(val).strip().replace(',', '')
    match = re.search(r'(\d+\.?\d*)', s)
    if match:
        try: return float(match.group(1))
        except: return 0
    return 0

def clean_text(val):
    if pd.isna(val): return ""
    return str(val).strip().replace('　', '').replace(' ', '')

# --- 2. メイン処理 ---
st.title("🏇 配置予測システム（重複エラー対策済）")

up_file = st.sidebar.file_uploader("当日配置表(Excel/CSV)", type=['xlsx', 'csv'])

if up_file:
    try:
        if up_file.name.endswith('.xlsx'):
            xl = pd.ExcelFile(up_file)
            sheet = st.sidebar.selectbox("シート選択", xl.sheet_names)
            df_raw = pd.read_excel(up_file, sheet_name=sheet, header=None) # 最初はヘッダーなしで読み込み
        else:
            try: df_raw = pd.read_csv(up_file, encoding='utf-8', header=None)
            except: df_raw = pd.read_csv(up_file, encoding='cp932', header=None)
        
        # 読み込み直後に列名を仮設定して重複回避
        df_raw.columns = make_columns_unique([f"Col_{i}" for i in range(len(df_raw.columns))])
    except Exception as e:
        st.error(f"読込エラー: {e}")
        st.stop()

    st.write("### 1. 読み込み範囲（ヘッダー行）の調整")
    header_row = st.number_input("項目名（場名, R, 馬名など）がある行番号 (0から開始)", 
                                 min_value=0, max_value=len(df_raw)-1, value=0)
    
    # 指定行をヘッダーにする
    df = df_raw.iloc[header_row:].reset_index(drop=True)
    if not df.empty:
        # 重複を排除して列名を設定
        df.columns = make_columns_unique(df.iloc[0])
        df = df.iloc[1:].reset_index(drop=True)
    
    st.write("現在の読み込み状態（最初の3行）:")
    st.dataframe(df.head(3))
    
    # --- 項目マッピング ---
    st.sidebar.header("🔍 2. 項目設定")
    all_cols = list(df.columns)
    
    def find_col(keywords, cols):
        for c in cols:
            if any(k in str(c) for k in keywords): return c
        return cols[0] if cols else ""

    col_r = st.sidebar.selectbox("「R（レース）」列", all_cols, index=all_cols.index(find_col(['R', 'Ｒ', 'レース'], all_cols)))
    col_place = st.sidebar.selectbox("「会場（場所）」列", all_cols, index=all_cols.index(find_col(['場所', '場名', '会場'], all_cols)))
    col_num = st.sidebar.selectbox("「馬番」列", all_cols, index=all_cols.index(find_col(['番', '馬番', '正番'], all_cols)))
    col_name = st.sidebar.selectbox("「馬名」列", all_cols, index=all_cols.index(find_col(['馬名', '馬'], all_cols)))
    col_odds = st.sidebar.selectbox("「単勝オッズ」列", all_cols, index=all_cols.index(find_col(['オッズ', '単勝'], all_cols)))
    
    col_jockey = st.sidebar.selectbox("「騎手」列", all_cols, index=all_cols.index(find_col(['騎手'], all_cols)))
    col_trainer = st.sidebar.selectbox("「厩舎」列", all_cols, index=all_cols.index(find_col(['厩舎', '調教師'], all_cols)))

    if st.sidebar.button("🚀 3. 解析を実行"):
        work_df = df.copy()
        
        # データのクリーニング
        work_df['R'] = work_df[col_r].apply(clean_numeric).astype(int)
        work_df['場名'] = work_df[col_place].apply(clean_text)
        work_df['正番'] = work_df[col_num].apply(clean_numeric).astype(int)
        work_df['馬名'] = work_df[col_name].apply(clean_text)
        work_df['単ｵｯｽﾞ'] = work_df[col_odds].apply(clean_numeric)
        
        # 1R以上の有効データのみ
        work_df = work_df[work_df['R'] > 0].copy()
        
        if work_df.empty:
            st.error("有効なデータがありません。ヘッダー行の設定が正しいか確認してください。")
        else:
            work_df['青塗フラグ'] = 0
            work_df['総合スコア'] = 0.0

            # 青塗判定
            for c_name in [col_jockey, col_trainer]:
                if c_name in work_df.columns:
                    mask = (work_df[c_name].astype(str).str.strip() != "")
                    dup = work_df[mask].duplicated(subset=['場名', c_name], keep=False)
                    work_df.loc[work_df.index[mask][dup], '青塗フラグ'] = 1
                    work_df.loc[work_df.index[mask][dup], '総合スコア'] += 9.0

            # AI予測
            if model:
                try:
                    X = pd.DataFrame({
                        '単ｵｯｽﾞ': work_df['単ｵｯｽﾞ'].replace(0, 99.0),
                        '青塗フラグ': work_df['青塗フラグ'],
                        'ペアフラグ': 0, '前日配置フラグ': 0
                    }).fillna(0)
                    probs = model.predict_proba(X)
                    work_df['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
                except: work_df['AI激走確率'] = 0.0
            
            st.session_state['result_df'] = work_df
            st.success("解析完了！")

    # --- 3. 表示 ---
    if 'result_df' in st.session_state:
        res_df = st.session_state['result_df']
        places = sorted([p for p in res_df['場名'].unique() if p != ""])
        if places:
            st.write("---")
            col_p, col_r_sel = st.columns(2)
            with col_p: target_place = st.selectbox("会場選択", places)
            with col_r_sel:
                r_list = sorted(res_df[res_df['場名'] == target_place]['R'].unique())
                target_r = st.selectbox("レース選択", r_list)
            
            view = res_df[(res_df['場名'] == target_place) & (res_df['R'] == target_r)].sort_values('AI激走確率', ascending=False)
            st.dataframe(
                view[['正番', '馬名', '単ｵｯｽﾞ', 'AI激走確率', '総合スコア', '青塗フラグ']],
                column_config={"AI激走確率": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)},
                use_container_width=True, hide_index=True
            )
else:
    st.info("👈 サイドバーからファイルをアップロードしてください。")
