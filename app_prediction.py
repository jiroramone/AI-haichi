import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os

# --- 1. 基本設定 ---
st.set_page_config(page_title="配置予測・不具合修正版", layout="wide")

@st.cache_resource
def load_ai_model():
    if os.path.exists('model.pkl'):
        try:
            with open('model.pkl', 'rb') as f:
                return pickle.load(f)
        except: return None
    return None

model = load_ai_model()

# --- 2. 徹底的な数値抽出関数 ---
def clean_numeric(val):
    if pd.isna(val): return 0
    s = str(val).strip().replace(',', '')
    # 数字（少数含む）を抽出
    match = re.search(r'(\d+\.?\d*)', s)
    if match:
        try:
            return float(match.group(1))
        except:
            return 0
    return 0

def clean_text(val):
    if pd.isna(val): return ""
    return str(val).strip().replace('　', '').replace(' ', '')

# --- 3. メイン処理 ---
st.title("🏇 配置予測システム（全項目0回避版）")

up_file = st.sidebar.file_uploader("当日配置表(Excel/CSV)", type=['xlsx', 'csv'])

if up_file:
    # 読み込み
    try:
        if up_file.name.endswith('.xlsx'):
            xl = pd.ExcelFile(up_file)
            sheet = st.sidebar.selectbox("シート選択", xl.sheet_names)
            df_raw = pd.read_excel(up_file, sheet_name=sheet)
        else:
            try: df_raw = pd.read_csv(up_file, encoding='utf-8')
            except: df_raw = pd.read_csv(up_file, encoding='cp932')
    except Exception as e:
        st.error(f"読込エラー: {e}")
        st.stop()

    # --- ステップ1: ヘッダー位置の調整 ---
    st.write("### 1. 読み込み範囲の確認")
    st.write("データが正しく表示（1行目が項目名に）されていない場合は、下の数字を調整してください。")
    header_row = st.number_input("項目名（ヘッダー）がある行番号 (0から開始)", min_value=0, max_value=len(df_raw)-1, value=0)
    
    # 指定行をヘッダーとして読み込み直し
    df = df_raw.iloc[header_row:].reset_index(drop=True)
    if not df.empty:
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
    
    st.write("現在の読み込み状態（最初の5行）:")
    st.dataframe(df.head(5))
    
    # --- ステップ2: 項目マッピング ---
    st.sidebar.header("🔍 2. 項目設定")
    all_cols = [str(c) for c in df.columns]
    
    def find_col(keywords, cols):
        for c in cols:
            if any(k in str(c) for k in keywords): return c
        return cols[0] if cols else ""

    col_r = st.sidebar.selectbox("「R（レース）」列", all_cols, index=all_cols.index(find_col(['R', 'Ｒ', 'レース'], all_cols)))
    col_place = st.sidebar.selectbox("「会場（場所）」列", all_cols, index=all_cols.index(find_col(['場所', '場名', '会場'], all_cols)))
    col_num = st.sidebar.selectbox("「馬番」列", all_cols, index=all_cols.index(find_col(['番', '馬番', '正番'], all_cols)))
    col_name = st.sidebar.selectbox("「馬名」列", all_cols, index=all_cols.index(find_col(['馬名', '馬'], all_cols)))
    col_odds = st.sidebar.selectbox("「単勝オッズ」列", all_cols, index=all_cols.index(find_col(['オッズ', '単勝'], all_cols)))
    
    # 配置判定用の追加列
    col_jockey = st.sidebar.selectbox("「騎手」列", all_cols, index=all_cols.index(find_col(['騎手'], all_cols)))
    col_trainer = st.sidebar.selectbox("「厩舎」列", all_cols, index=all_cols.index(find_col(['厩舎', '調教師'], all_cols)))

    if st.sidebar.button("🚀 3. 解析を実行"):
        work_df = df.copy()
        
        # クリーニング実行（ここで0や99になるのを防ぐ）
        work_df['R'] = work_df[col_r].apply(clean_numeric).astype(int)
        work_df['場名'] = work_df[col_place].apply(clean_text)
        work_df['正番'] = work_df[col_num].apply(clean_numeric).astype(int)
        work_df['馬名'] = work_df[col_name].apply(clean_text)
        work_df['単ｵｯｽﾞ'] = work_df[col_odds].apply(clean_numeric)
        
        # 有効なデータ（1R以上）のみ抽出
        work_df = work_df[work_df['R'] > 0].copy()
        
        if work_df.empty:
            st.error("有効なデータが見つかりませんでした。ヘッダー行の設定や列の選択が正しいか確認してください。")
        else:
            # フラグ計算
            work_df['青塗フラグ'] = 0
            work_df['総合スコア'] = 0.0

            # 簡易青塗判定
            for c_name in [col_jockey, col_trainer]:
                if c_name in work_df.columns:
                    mask = (work_df[c_name].astype(str).str.strip() != "") & (work_df[c_name].notna())
                    dup = work_df[mask].duplicated(subset=['場名', c_name], keep=False)
                    work_df.loc[work_df.index[mask][dup], '青塗フラグ'] = 1
                    work_df.loc[work_df.index[mask][dup], '総合スコア'] += 9.0

            # AI予測
            if model:
                try:
                    X = pd.DataFrame({
                        '単ｵｯｽﾞ': work_df['単ｵｯｽﾞ'].replace(0, 99.0),
                        '青塗フラグ': work_df['青塗フラグ'],
                        'ペアフラグ': 0, # 未実装分は0
                        '前日配置フラグ': 0
                    }).fillna(0)
                    
                    probs = model.predict_proba(X)
                    work_df['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
                except Exception as e:
                    st.warning(f"AI予測エラー: {e}")
                    work_df['AI激走確率'] = 0.0
            
            st.session_state['result_df'] = work_df
            st.success("解析が完了しました！下で結果を確認してください。")

    # --- 4. 結果表示 ---
    if 'result_df' in st.session_state:
        res_df = st.session_state['result_df']
        st.write("---")
        
        places = sorted([p for p in res_df['場名'].unique() if p != ""])
        if places:
            col_p, col_r = st.columns(2)
            with col_p: target_place = st.selectbox("会場選択", places)
            with col_r:
                r_list = sorted(res_df[res_df['場名'] == target_place]['R'].unique())
                target_r = st.selectbox("レース選択", r_list)
            
            view = res_df[(res_df['場名'] == target_place) & (res_df['R'] == target_r)].sort_values('AI激走確率', ascending=False)
            
            st.dataframe(
                view[['正番', '馬名', '単ｵｯｽﾞ', 'AI激走確率', '総合スコア', '青塗フラグ']],
                column_config={
                    "AI激走確率": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                    "単ｵｯｽﾞ": st.column_config.NumberColumn(format="%.1f")
                },
                use_container_width=True, hide_index=True
            )
