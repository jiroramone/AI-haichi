import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os
import plotly.express as px

# --- 1. 基本設定 ---
st.set_page_config(page_title="AI配置分析システム Ver.Final", layout="wide")

@st.cache_resource
def load_ai_model():
    if os.path.exists('model.pkl'):
        try:
            with open('model.pkl', 'rb') as f:
                return pickle.load(f)
        except: return None
    return None

model = load_ai_model()

def to_half_width(text):
    if pd.isna(text): return ""
    table = str.maketrans('０１２３４５６７８９．', '0123456789.')
    return re.sub(r'[^\d\.]', '', str(text).translate(table))

def clean_num(val):
    s = to_half_width(val)
    match = re.search(r'(\d+\.?\d*)', s)
    return float(match.group(1)) if match else 0.0

# --- 2. データ解析エンジン（属性列の確保・F列固定） ---
def build_clean_data(file):
    try:
        # ファイル読み込み
        if file.name.endswith('.xlsx'):
            raw = pd.read_excel(file, header=None, engine='openpyxl')
        else:
            try: raw = pd.read_csv(file, header=None, encoding='utf-8')
            except: raw = pd.read_csv(file, header=None, encoding='cp932')
        
        # ヘッダー行探索
        best_row, max_hits = 0, 0
        keywords = ['場所', 'Ｒ', '馬名', '正番', '騎手']
        for i in range(min(len(raw), 30)):
            row_vals = [str(x) for x in raw.iloc[i].values]
            hits = sum(1 for k in keywords if any(k in v for v in row_vals))
            if hits > max_hits:
                max_hits, best_row = hits, i
        
        headers = [str(x).strip() for x in raw.iloc[best_row].values]
        data_body = raw.iloc[best_row+1:].reset_index(drop=True)
        
        # 列特定（F列=正番 固定）
        col_idx = {}
        mapping_rules = {
            '場名': ['場所', '場名'], 'R': ['Ｒ', 'R', 'レース'],
            '正番': ['正番', '馬番'], '馬名': ['馬名'],
            '単ｵｯｽﾞ': ['単オッズ', '単ｵｯｽﾞ', 'オッズ'],
            '騎手': ['騎手'], '調教師': ['調教師', '厩舎'], '馬主': ['馬主'],
            '着順': ['着順', '着', '結果']
        }
        
        for internal, keys in mapping_rules.items():
            for idx, h_name in enumerate(headers):
                if internal == '正番': 
                    # 正番は「逆」や「枠」を含まない列を優先、なければF列(5)
                    if any(k in h_name for k in keys) and '逆' not in h_name and '枠' not in h_name:
                        col_idx[internal] = idx; break
                elif any(k in h_name for k in keys):
                    col_idx[internal] = idx; break
        
        if '正番' not in col_idx and len(data_body.columns) > 5:
            col_idx['正番'] = 5

        # データフレーム構築
        df = pd.DataFrame(index=data_body.index)
        for col in ['R', '正番', '場名', '馬名', '単ｵｯｽﾞ', '騎手', '調教師', '馬主', '着順']:
            idx = col_idx.get(col)
            if idx is not None: df[col] = data_body.iloc[:, idx]
            else: df[col] = np.nan if col == '着順' else (0 if col in ['R','正番'] else (99.0 if col == '単ｵｯｽﾞ' else "不明"))

        df['R'] = df['R'].apply(clean_num).astype(int)
        df['正番'] = df['正番'].apply(clean_num).astype(int)
        df['単ｵｯｽﾞ'] = df['単ｵｯｽﾞ'].apply(clean_num).replace(0, 99.0)
        df = df[df['R'] > 0].copy()

        # --- 配置・ペア判定 ---
        # 【重要】ここで必ず初期化する（KeyError防止）
        df['青塗フラグ'] = 0; df['ペアフラグ'] = 0; df['判定理由'] = ""; df['属性'] = ""
        
        df['頭数'] = df.groupby(['場名', 'R'])['正番'].transform('max').fillna(16)
        df['逆番'] = (df['頭数'] + 1) - df['正番']
        df['正循環'] = df['頭数'] + df['正番']
        df['逆循環'] = df['頭数'] + df['逆番']

        for col in ['騎手', '調教師', '馬主']:
            for name, group in df.groupby(['場名', col] if col=='騎手' else col):
                if len(group) < 2 or str(name) in ['nan', '不明', '']: continue
                
                rows = group.sort_values('R').to_dict('records')
                indices = group.sort_values('R').index
                
                for i in range(len(rows)):
                    curr = rows[i]
                    curr_sets = {curr['正番'], curr['逆番'], curr['正循環'], curr['逆循環']}
                    
                    found_type = None # None, 'pair', 'blue'
                    
                    for j in range(len(rows)):
                        if i == j: continue
                        target = rows[j]
                        target_sets = {target['正番'], target['逆番'], target['正循環'], target['逆循環']}
                        
                        if curr_sets.intersection(target_sets):
                            # 隣接レースならペア、それ以外は青塗
                            if abs(curr['R'] - target['R']) == 1:
                                found_type = 'pair'
                                break # ペアが見つかれば優先してループを抜ける
                            else:
                                found_type = 'blue'
                    
                    idx = indices[i]
                    # 属性列に追記
                    current_attr = f"{col}:{name}"
                    if current_attr not in str(df.at[idx, '属性']):
                        df.at[idx, '属性'] += f"{current_attr} / "

                    if found_type == 'pair':
                        df.at[idx, 'ペアフラグ'] = 1
                        if f"★{col}ペア" not in df.at[idx, '判定理由']:
                            df.at[idx, '判定理由'] += f"★{col}ペア "
                    elif found_type == 'blue':
                        df.at[idx, '青塗フラグ'] = 1
                        if f"★{col}青塗" not in df.at[idx, '判定理由']:
                            df.at[idx, '判定理由'] += f"★{col}青塗 "

        # AI基礎予測
        df['AI激走確率'] = 0.0
        if model:
            try:
                X = df[['単ｵｯｽﾞ', '青塗フラグ']].copy()
                X['ペアフラグ'] = df['ペアフラグ']
                X['前日フラグ'] = 0
                df['AI激走確率'] = [round(p[1] * 100, 1) for p in model.predict_proba(X)]
            except: pass

        return df, "success"
    except Exception as e:
        import traceback
        return pd.DataFrame(), traceback.format_exc()

# --- 3. 連動ロジック（エラーガード付き） ---
def apply_pair_link_logic(df):
    # 安全装置: 属性列がない場合は作成
    if '属性' not in df.columns: df['属性'] = ""
    if '判定理由' not in df.columns: df['判定理由'] = ""
    
    df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
    df['ペア連動ボーナス'] = 0.0
    
    # 1. ペア的中の特定
    hit_pair_keys = set()
    hits = df[df['着順'] <= 3]
    
    for _, row in hits.iterrows():
        # 安全な文字列変換
        reasons = str(row.get('判定理由', ''))
        attrs = str(row.get('属性', ''))
        
        if "ペア" in reasons:
            for attr in attrs.split(' / '):
                if attr.strip():
                    hit_pair_keys.add((row['場名'], row['R'], attr.strip()))

    # 2. 次レースへの反映
    for idx, row in df.iterrows():
        reasons = str(row.get('判定理由', ''))
        attrs = str(row.get('属性', ''))
        
        if "ペア" in reasons:
            for attr in attrs.split(' / '):
                target_attr = attr.strip()
                if not target_attr: continue
                
                # 「前のレース(R-1)」で「同じ属性」のペアが来ていればボーナス
                if (row['場名'], row['R'] - 1, target_attr) in hit_pair_keys:
                    df.at[idx, 'ペア連動ボーナス'] = 20.0
                    # 連動理由を追記（可視化のため）
                    if "🚀連動" not in reasons:
                        df.at[idx, '判定理由'] = "🚀連動 " + df.at[idx, '判定理由']
                    break

    df['最終期待値'] = df['AI激走確率'] + df['ペア連動ボーナス']
    return df

# --- 4. UI ---
st.title("🎯 AI配置分析：ペア連動完全版")

up_file = st.sidebar.file_uploader("当日配置表をアップロード", type=['xlsx', 'csv'])

# セッション状態の初期化チェック（列不足エラーを防ぐ）
if up_file:
    # ファイルが変わった、またはデータ未ロード時に解析
    if 'main_df' not in st.session_state or st.sidebar.button("🔄 データを再解析（リセット）"):
        res, status = build_clean_data(up_file)
        if status == "success":
            st.session_state['main_df'] = res
        else:
            st.error("解析失敗"); st.code(status)

    if 'main_df' in st.session_state:
        # ここでコピーしてロジックを適用
        try:
            df = apply_pair_link_logic(st.session_state['main_df'].copy())
            
            places = sorted([p for p in df['場名'].unique() if str(p) not in ['nan','不明']])
            if places:
                target_p = st.sidebar.selectbox("会場を選択", places)
                r_list = sorted(df[df['場名'] == target_p]['R'].unique())
                target_r = st.sidebar.selectbox("レースを選択", r_list)

                view = df[(df['場名'] == target_p) & (df['R'] == target_r)].sort_values('最終期待値', ascending=False)
                st.subheader(f"📊 {target_p} {int(target_r)}R 激走予測")
                
                # トップ馬の強調
                top = view.iloc[0]
                if top['ペア連動ボーナス'] > 0:
                    st.success(f"🚀 ペア連動発生中！ {int(top['正番'])}番 {top['馬名']} (期待値+20)")

                # テーブル表示
                cols = ['正番', '馬名', '判定理由', 'AI激走確率', 'ペア連動ボーナス', '最終期待値', '着順']
                final_cols = [c for c in cols if c in view.columns]
                
                ed = st.data_editor(view[final_cols], key=f"ed_{target_p}_{target_r}", 
                                    hide_index=True, use_container_width=True,
                                    column_config={
                                        "最終期待値": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
                                        "正番": st.column_config.NumberColumn(format="%d")
                                    })
                
                if st.button("🔄 着順を保存して連動を反映"):
                    for _, row in ed.iterrows():
                        st.session_state['main_df'].loc[(st.session_state['main_df']['場名']==target_p) & 
                                                (st.session_state['main_df']['R']==target_r) & 
                                                (st.session_state['main_df']['正番']==row['正番']), '着順'] = row['着順']
                    st.rerun()
            else:
                st.warning("表示可能なレースデータがありません。")

        except Exception as e:
            st.error(f"処理エラー: {e}")
            if st.button("データをリセットして修復"):
                del st.session_state['main_df']
                st.rerun()
else:
    st.info("左側のサイドバーからファイルを読み込んでください。")
