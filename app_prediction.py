import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os
import plotly.express as px

# --- 1. 基本設定 ---
st.set_page_config(page_title="AI配置馬券・統合分析システム", layout="wide")

@st.cache_resource
def load_ai_model():
    if os.path.exists('model.pkl'):
        try:
            with open('model.pkl', 'rb') as f:
                return pickle.load(f)
        except: return None
    return None

model = load_ai_model()

# --- 2. データクレンジング（指示通りF列を正番固定） ---
def to_half_width(text):
    if pd.isna(text): return ""
    table = str.maketrans('０１２３４５６７８９．', '0123456789.')
    return re.sub(r'[^\d\.]', '', str(text).translate(table))

def normalize_name(x):
    if pd.isna(x): return ''
    s = str(x).strip().replace('　', '').replace(' ', '')
    s = re.split(r'[,(（/]', s)[0]
    return re.sub(r'[★☆▲△◇$*]', '', s)

def load_data_robust(file):
    try:
        if file.name.endswith('.xlsx'):
            raw = pd.read_excel(file, header=None, engine='openpyxl')
        else:
            try: raw = pd.read_csv(file, header=None, encoding='utf-8')
            except: raw = pd.read_csv(file, header=None, encoding='cp932')
        
        # ヘッダー行探索
        best_row, max_hits = 0, 0
        keywords = ['場所', 'Ｒ', '馬名', '正番']
        for i in range(min(len(raw), 25)):
            row_vals = [str(x) for x in raw.iloc[i].values]
            hits = sum(1 for k in keywords if any(k in v for v in row_vals))
            if hits > max_hits:
                max_hits, best_row = hits, i
        
        headers = [str(x).strip() for x in raw.iloc[best_row].values]
        # F列(6番目)を強制的に「正番」にする
        if len(headers) >= 6: headers[5] = "正番"
        
        df_data = raw.iloc[best_row+1:].reset_index(drop=True)
        
        # 列名マッピング
        mapping = {
            '場名': ['場所', '場名', '会場'],
            'R': ['Ｒ', 'R', 'レース'],
            '馬名': ['馬名'],
            '単ｵｯｽﾞ': ['単オッズ', '単ｵｯｽﾞ', 'オッズ'],
            '騎手': ['騎手'], '厩舎': ['調教師', '厩舎'], '馬主': ['馬主'],
            '着順': ['着順', '着']
        }
        
        col_map = {}
        for internal, keys in mapping.items():
            for idx, h_name in enumerate(headers):
                if any(k in h_name for k in keys):
                    col_map[idx] = internal; break
        # 正番(F列)は固定
        col_map[5] = '正番'

        df = pd.DataFrame()
        for idx, name in col_map.items():
            if idx < len(df_data.columns): df[name] = df_data.iloc[:, idx]
        
        # 型変換
        def clean_num(val):
            match = re.search(r'(\d+\.?\d*)', to_half_width(val))
            return float(match.group(1)) if match else 0.0

        df['R'] = df['R'].apply(clean_num).astype(int)
        df['正番'] = df['正番'].apply(clean_num).astype(int)
        df['単ｵｯｽﾞ'] = df['単ｵｯｽﾞ'].apply(clean_num).replace(0, 99.0)
        for col in ['騎手', '厩舎', '馬主', '馬名', '場名']:
            if col in df.columns: df[col] = df[col].apply(normalize_name)

        return df[df['R'] > 0].copy(), "success"
    except Exception as e:
        import traceback
        return pd.DataFrame(), traceback.format_exc()

# --- 3. 配置計算エンジン（ご提示いただいたロジックを完全移植） ---
def analyze_haichi_ai(df, df_prev=None):
    if df.empty: return df
    
    # 基本数字の算出
    df['頭数'] = df.groupby(['場名', 'R'])['正番'].transform('max').fillna(16).astype(int)
    df['逆番'] = (df['頭数'] + 1) - df['正番']
    df['正循環'] = df['頭数'] + df['正番']
    df['逆循環'] = df['頭数'] + df['逆番']
    
    df['タイプ_list'] = [[] for _ in range(len(df))]
    df['属性_list'] = [[] for _ in range(len(df))]
    df['パターン_list'] = [[] for _ in range(len(df))]
    df['スコア'] = 0.0
    
    idx_map = {(row['場名'], row['R'], row['正番']): idx for idx, row in df.iterrows()}

    # ① 青塗・青隣判定
    blue_info = []
    for col in ['騎手', '厩舎', '馬主']:
        if col not in df.columns: continue
        g_keys = ['場名', col] if col == '騎手' else [col]
        for name, group in df.groupby(g_keys):
            if len(group) < 2 or not name or name == '不明': continue
            all_sets = [{r['正番'], r['逆番'], r['正循環'], r['逆循環']} for _, r in group.iterrows()]
            common = set.intersection(*all_sets)
            if common:
                for _, row in group.iterrows():
                    idx = idx_map.get((row['場名'], row['R'], row['正番']))
                    if idx is not None:
                        df.at[idx, 'タイプ_list'].append(f'★{col}青塗')
                        df.at[idx, '属性_list'].append(f'{col}:{name}')
                        df.at[idx, 'パターン_list'].append('青塗')
                        df.at[idx, 'スコア'] += 9.2 if col == '騎手' else 9.0

    # ② ペア判定 (連続レース)
    pair_labels = list("ABCDEFGHIJKLMNOP")
    for col in ['騎手', '厩舎', '馬主']:
        if col not in df.columns: continue
        for name, group in df.groupby(['場名', col] if col=='騎手' else col):
            if len(group) < 2 or not name: continue
            rows = group.sort_values('R').to_dict('records')
            for i in range(len(rows)-1):
                r1, r2 = rows[i], rows[i+1]
                if r2['R'] == r1['R'] + 1: # 連続レース
                    v1 = [r1['正番'], r1['逆番'], r1['正循環'], r1['逆循環']]
                    v2 = [r2['正番'], r2['逆番'], r2['正循環'], r2['逆循環']]
                    pats = [pair_labels[x*4+y] for x in range(4) for y in range(4) if v1[x]==v2[y] and v1[x]!=0]
                    if pats:
                        is_c = any(x in pats for x in ['C','D','G','H'])
                        for r_data in [r1, r2]:
                            idx = idx_map.get((r_data['場名'], r_data['R'], r_data['正番']))
                            if idx is not None:
                                df.at[idx, 'タイプ_list'].append('◎ペア' if is_c else '○ペア')
                                df.at[idx, '属性_list'].append(f'{col}:{name}')
                                df.at[idx, 'パターン_list'].append(",".join(pats))
                                df.at[idx, 'スコア'] += 4.5 if is_c else 3.5

    # AI基礎確率の算出
    if model:
        try:
            X = pd.DataFrame({
                '単ｵｯｽﾞ': df['単ｵｯｽﾞ'],
                '青塗フラグ': df['タイプ_list'].apply(lambda x: 1 if any('青塗' in str(v) for v in x) else 0),
                'ペアフラグ': df['タイプ_list'].apply(lambda x: 1 if any('ペア' in str(v) for v in x) else 0),
                '前日フラグ': 0
            })
            df['AI激走確率'] = [round(p[1] * 100, 1) for p in model.predict_proba(X)]
        except: df['AI激走確率'] = 0.0
    else: df['AI激走確率'] = 0.0

    df['タイプ'] = df['タイプ_list'].apply(lambda x: ' / '.join(x) if x else '無')
    df['属性'] = df['属性_list'].apply(lambda x: ' / '.join(list(set(x))) if x else '')
    df['パターン'] = df['パターン_list'].apply(lambda x: ','.join(x) if x else '')
    return df

# --- 4. 判定ロジック（エネルギー状態・バイアス反映） ---
def apply_ai_ranking(df):
    if df.empty: return df
    df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
    
    # 今日の的中傾向（属性・パターンのバイアス）
    hit_results = df[df['着順'] <= 3]
    hit_attrs = set([a for _, row in hit_results.iterrows() for a in str(row.get('属性', '')).split(' / ')])
    hit_pats = set([p for pats in hit_results['パターン'].dropna() for p in str(pats).split(',') if p])

    def get_ai_metrics(row):
        base_score = row.get('スコア', 0)
        ai_prob = row.get('AI激走確率', 0)
        p_list = str(row.get('パターン', '')).split(',')
        
        # バイアス加点（今日の流れに乗っているか）
        bias_bonus = 5.0 if any(p in hit_pats and len(p)==1 for p in p_list) else 0.0
        
        # 的中済み属性へのペナルティ/エネルギー消費（エネルギー状態）
        reasons = []
        for ra in str(row.get('属性', '')).split(' / '):
            if ra in hit_attrs: reasons.append(f"{ra.split(':')[0] if ':' in ra else '本人'}好走済")
        
        energy_penalty = -3.5 if reasons else 0.0
        
        # 最終スコア合算
        total = base_score + (ai_prob / 5.0) + bias_bonus + energy_penalty
        total -= (25.0 if row.get('単ｵｯｽﾞ', 0) > 50 else 0) # 大穴ペナルティ
        
        # 推奨区分
        if total >= 25: rec = "👑 盤石の軸"
        elif total >= 18: rec = "✨ 推奨軸"
        elif total >= 12: rec = "🔥 激熱相手"
        elif base_score > 0: rec = "▲ 配置注目"
        else: rec = ""
        
        return pd.Series([total, f"⚠️{','.join(set(reasons))}" if reasons else "良好", rec])

    df[['総合スコア', 'エネルギー状態', '推奨買い目']] = df.apply(get_ai_metrics, axis=1)
    return df

# --- 5. UI ---
st.title("🏇 AI配置馬券分析システム")

with st.sidebar:
    st.header("📂 データ読込")
    up_curr = st.file_uploader("当日配置表", type=['xlsx', 'csv'], key="curr")
    if up_curr and 'analyzed_df' in st.session_state:
        st.divider(); st.header("💾 保存")
        csv = st.session_state['analyzed_df'].to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 分析CSVを保存", csv, f"progress_{up_curr.name}.csv")

if up_curr:
    df_raw, status = load_data_robust(up_curr)
    if status == "success":
        if 'analyzed_df' not in st.session_state:
            st.session_state['analyzed_df'] = apply_ai_ranking(analyze_haichi_ai(df_raw))
        
        full_df = st.session_state['analyzed_df']

        # ① 結果入力（流動的なバイアス計算のため）
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
                            disp = race_full[race_full['スコア'] > 0].copy()
                            if disp.empty: 
                                st.caption("配置該当なし")
                                edited_dfs.append(race_full)
                            else:
                                target_cols = ['正番','馬名','単ｵｯｽﾞ','AI激走確率','判定理由','エネルギー状態','総合スコア','着順','推奨買い目']
                                # 存在する列だけを表示
                                view_cols = [c for c in target_cols if c in disp.columns]
                                if '判定理由' not in disp.columns: disp['判定理由'] = disp['タイプ']
                                
                                ed = st.data_editor(disp[view_cols], hide_index=True, use_container_width=True, key=f"ed_{place}_{r_num}",
                                                    column_config={"AI激走確率": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})
                                updated = race_full.copy()
                                for _, row in ed.iterrows():
                                    updated.loc[updated['正番'] == row['正番'], '着順'] = row['着順']
                                edited_dfs.append(updated)
            
            if st.form_submit_button("🔄 入力を確定して全体を更新"):
                st.session_state['analyzed_df'] = apply_ai_ranking(pd.concat(edited_dfs, ignore_index=True))
                st.rerun()

        # ② AI特選推奨
        st.divider(); st.subheader("👑 AI特選推奨馬")
        future_df = full_df[(full_df['着順'].isna()) & (full_df['総合スコア'] >= 15)].sort_values('総合スコア', ascending=False)
        if not future_df.empty:
            st.dataframe(future_df[['場名','R','正番','馬名','単ｵｯｽﾞ','タイプ','エネルギー状態','総合スコア','推奨買い目']], use_container_width=True, hide_index=True)
        else:
            st.info("現在、基準を超える推奨馬はいません。結果を入力するとバイアスで評価が上がる場合があります。")

else:
    st.info("左側のサイドバーからファイルを読み込んでください。")
