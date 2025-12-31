import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os

# --- 1. 基本設定 ---
st.set_page_config(page_title="配置予測・表示改善版", layout="wide")

@st.cache_resource
def load_ai_model():
    if os.path.exists('model.pkl'):
        try:
            with open('model.pkl', 'rb') as f:
                return pickle.load(f)
        except: return None
    return None

model = load_ai_model()

# --- 2. 徹底的なクリーニング関数 ---
def clean_numeric(val):
    """文字混じりの数字から数字だけを抜き出す (例: '1R' -> 1)"""
    s = str(val).strip()
    match = re.search(r'(\d+)', s)
    return int(match.group(1)) if match else 0

def clean_text(val):
    """テキストから空白や特殊文字を消す"""
    if pd.isna(val): return ""
    return str(val).strip().replace('　', '').replace(' ', '')

# --- 3. メイン処理 ---
st.title("🏇 配置予測システム（表示修正版）")

up_file = st.sidebar.file_uploader("当日配置表(Excel/CSV)", type=['xlsx', 'csv'])

if up_file:
    # 読み込み
    try:
        if up_file.name.endswith('.xlsx'):
            df = pd.read_excel(up_file, engine='openpyxl')
        else:
            try: df = pd.read_csv(up_file, encoding='utf-8')
            except: df = pd.read_csv(up_file, encoding='cp932')
    except Exception as e:
        st.error(f"読込エラー: {e}")
        st.stop()
    
    # 項目設定
    st.sidebar.header("🔍 項目設定")
    all_cols = [str(c) for c in df.columns]
    
    def find_col(keywords, cols):
        for c in cols:
            if any(k in str(c) for k in keywords): return c
        return cols[0]

    col_r = st.sidebar.selectbox("「R」はどの列？", all_cols, index=all_cols.index(find_col(['R', 'Ｒ', 'レース'], all_cols)))
    col_place = st.sidebar.selectbox("「会場」はどの列？", all_cols, index=all_cols.index(find_col(['場所', '場名', '会場'], all_cols)))
    col_num = st.sidebar.selectbox("「馬番」はどの列？", all_cols, index=all_cols.index(find_col(['番', '馬番', '正番'], all_cols)))
    col_name = st.sidebar.selectbox("「馬名」はどの列？", all_cols, index=all_cols.index(find_col(['馬名', '馬', '名称'], all_cols)))
    col_odds = st.sidebar.selectbox("「オッズ」はどの列？", all_cols, index=all_cols.index(find_col(['オッズ', '単勝'], all_cols)))

    if st.sidebar.button("🚀 解析を実行"):
        # 必要な列だけをコピーして名寄せ・クリーニング
        work_df = df.copy()
        work_df['R'] = work_df[col_r].apply(clean_numeric)
        work_df['場名'] = work_df[col_place].apply(clean_text)
        work_df['正番'] = work_df[col_num].apply(clean_numeric)
        work_df['馬名'] = work_df[col_name].apply(clean_text)
        work_df['単ｵｯｽﾞ'] = pd.to_numeric(work_df[col_odds], errors='coerce').fillna(99.0)
        
        # フラグ初期化
        work_df['青塗フラグ'] = 0; work_df['AI激走確率'] = 0.0
        
        # AI予測
        if model:
            try:
                # 学習時と同じ4項目（必要ならここで調整）
                X = pd.DataFrame({
                    '単ｵｯｽﾞ': work_df['単ｵｯｽﾞ'],
                    '青塗フラグ': 0, 'ペアフラグ': 0, '前日配置フラグ': 0 # ダミー
                })
                probs = model.predict_proba(X)
                work_df['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
            except: pass

        st.session_state['result_df'] = work_df
        st.success(f"✅ {len(work_df)}頭のデータを解析しました")

    # --- 表示部 ---
    if 'result_df' in st.session_state:
        res_df = st.session_state['result_df']
        
        # 会場とレースの選択
        places = sorted([p for p in res_df['場名'].unique() if p != ""])
        if not places:
            st.error("会場名が正しく読み込めていません。設定を見直してください。")
        else:
            target_place = st.selectbox("会場を選択", places)
            
            r_nums = sorted(res_df[res_df['場名'] == target_place]['R'].unique())
            if not r_nums:
                st.warning(f"「{target_place}」にはレースデータがありません。")
            else:
                target_r = st.selectbox("レースを選択", r_nums)
                
                # フィルタリング実行
                view = res_df[(res_df['場名'] == target_place) & (res_df['R'] == target_r)].copy()
                
                if view.empty:
                    st.warning("⚠️ 指定した条件に合う馬が見つかりませんでした。")
                    with st.expander("詳細な理由を確認"):
                        st.write(f"現在選択中の会場: '{target_place}'")
                        st.write(f"現在選択中のレース: {target_r}")
                        st.write("データにある会場一覧:", places)
                else:
                    st.subheader(f"📊 {target_place} {target_r}R 予測結果")
                    st.dataframe(
                        view[['正番', '馬名', '単ｵｯｽﾞ', 'AI激走確率']].sort_values('AI激走確率', ascending=False),
                        column_config={"AI激走確率": st.column_config.ProgressColumn(min_value=0, max_value=100)},
                        use_container_width=True, hide_index=True
                    )
else:
    st.info("👈 左のサイドバーからファイルを読み込んで「解析を実行」を押してください。")
