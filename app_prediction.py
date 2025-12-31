import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os

# --- 1. 基本設定 ---
st.set_page_config(page_title="AI配置判定システム", layout="wide")

@st.cache_resource
def load_ai_model():
    MODEL_PATH = 'model.pkl'
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"モデル読み込みエラー: {e}")
            return None
    return None

model = load_ai_model()

def to_half_width(text):
    if pd.isna(text): return text
    table = str.maketrans('０１２３４５６７８９．', '0123456789.')
    return re.sub(r'[^\d\.]', '', str(text).translate(table))

# --- 2. データ読み取り・配置解析エンジン ---
def analyze_data(file):
    try:
        # ファイル読み込み
        if file.name.endswith('.xlsx'):
            df_raw = pd.read_excel(file, header=None, engine='openpyxl')
        else:
            try: df_raw = pd.read_csv(file, header=None, encoding='utf-8')
            except: df_raw = pd.read_csv(file, header=None, encoding='cp932')

        # 見出し行（ヘッダー）を自動で見つける
        best_row, max_hits = 0, 0
        keywords = ['場所', 'R', '馬名', 'オッズ']
        for i in range(min(len(df_raw), 25)):
            row_str = "".join(df_raw.iloc[i].astype(str))
            hits = sum(1 for k in keywords if k in row_str)
            if hits > max_hits:
                max_hits, best_row = hits, i
        
        df = df_raw.iloc[best_row:].reset_index(drop=True)
        headers = [str(c).strip() for c in df.iloc[0]]
        
        # 【重要】F列（6列目）を強制的に「正番」にする
        if len(headers) >= 6:
            headers[5] = "正番"
        
        df.columns = headers
        df = df.iloc[1:].reset_index(drop=True)

        # 項目名の名寄せ（「場所」を「場名」へ）
        mapping = {
            '場名': ['場所', '場名', '競馬場'],
            'R': ['R', 'レース', '番組'],
            '馬名': ['馬名', '名称'],
            '単ｵｯｽﾞ': ['単ｵｯｽﾞ', 'オッズ', '単勝'],
            '騎手': ['騎手'], '厩舎': ['厩舎', '調教師'], '馬主': ['馬主']
        }
        col_map = {}
        for internal, keys in mapping.items():
            for c in df.columns:
                if any(k == str(c) for k in keys):
                    col_map[c] = internal; break
        df = df.rename(columns=col_map)

        # 数値のクリーンアップ（1Rなどの文字を数字にする）
        def extract_num(val):
            s = to_half_width(val)
            match = re.search(r'(\d+\.?\d*)', str(s))
            return float(match.group(1)) if match else 0.0

        for c in ['R', '正番', '単ｵｯｽﾞ']:
            if c in df.columns:
                df[c] = df[c].apply(extract_num)
            else:
                df[c] = 0.0 if c != '単ｵｯｽﾞ' else 99.0

        df = df[df['R'] > 0].copy()

        # --- 配置判定ロジック ---
        df['青塗フラグ'] = 0; df['判定'] = ""
        df['頭数'] = df.groupby(['場名', 'R'])['正番'].transform('max')
        df['逆番'] = (df['頭数'] + 1) - df['正番']
        df['正循環'] = df['頭数'] + df['正番']
        df['逆循環'] = df['頭数'] + df['逆番']
        
        idx_map = {(row['場名'], row['R'], int(row['正番'])): idx for idx, row in df.iterrows()}

        # 青塗（配置一致）の判定
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
                            df.at[idx, '判定'] += f"★{col}青塗 "

        # --- AI激走確率の計算 ---
        if model:
            try:
                # 学習済みモデルが期待する特徴量（順番と名前を合わせる）
                X = df[['単ｵｯｽﾞ', '青塗フラグ']].copy()
                X['ペアフラグ'] = 0; X['前日配置フラグ'] = 0 # 拡張用
                
                probs = model.predict_proba(X)
                df['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
            except Exception as e:
                st.sidebar.error(f"AI予測エラー: {e}")
                df['AI激走確率'] = 0.0
        else:
            df['AI激走確率'] = 0.0

        return df, "success"
    except Exception as e:
        return pd.DataFrame(), str(e)

# --- 3. UI画面 ---
st.title("🎯 AI配置馬券 激走判定システム")



file = st.sidebar.file_uploader("当日配置表をアップロード", type=['xlsx', 'csv'])

if file:
    df, status = analyze_data(file)
    
    if status == "success" and not df.empty:
        # 会場とレースの選択
        places = sorted([p for p in df['場名'].unique() if str(p) != 'nan'])
        
        if places:
            target_place = st.sidebar.selectbox("会場を選択", places)
            r_list = sorted(df[df['場名'] == target_place]['R'].unique())
            target_r = st.sidebar.selectbox("レースを選択", r_list)

            # 結果表示
            view = df[(df['場名'] == target_place) & (df['R'] == target_r)].sort_values('AI激走確率', ascending=False)
            
            st.subheader(f"📊 {target_place} {int(target_r)}R AI判定結果")
            
            # 推奨馬を大きく表示
            top = view.iloc[0]
            st.success(f"👑 AI最推奨: {int(top['正番'])}番 {top['馬名']} (激走確率 {top['AI激走確率']}%)")

            # テーブル表示
            st.dataframe(
                view[['正番', '馬名', '単ｵｯｽﾞ', 'AI激走確率', '判定']],
                column_config={
                    "AI激走確率": st.column_config.ProgressColumn("激走確率", format="%.1f%%", min_value=0, max_value=100),
                    "正番": st.column_config.NumberColumn("馬番", format="%d"),
                    "単ｵｯｽﾞ": st.column_config.NumberColumn("オッズ", format="%.1f")
                },
                use_container_width=True, hide_index=True
            )
        else:
            st.error("会場（場所）が特定できません。エクセル内の『場所』列を確認してください。")
    else:
        st.error(f"データの読み込みに失敗しました: {status}")
else:
    st.info("左側のサイドバーから配置表（エクセルなど）をアップロードしてください。")
    
