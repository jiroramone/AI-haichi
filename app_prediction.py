import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os
import plotly.express as px
import time

# --- 1. 基本設定 ---
st.set_page_config(page_title="配置馬券 AI分析統合システム", layout="wide")

# AIモデルの読み込み
MODEL_PATH = 'model.pkl'
model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except:
        st.error("model.pkl の読み込みに失敗しました。")

def to_half_width(text):
    if pd.isna(text): return text
    table = str.maketrans('０１２３４５６７８９．', '0123456789.')
    return re.sub(r'[^\d\.]', '', str(text).translate(table))

def normalize_name(x):
    if pd.isna(x): return ''
    s = str(x).strip().replace('　', '').replace(' ', '')
    s = re.split(r'[,(（/]', s)[0]
    return re.sub(r'[★☆▲△◇$*]', '', s)

# --- 2. データ読み込み ---
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        else:
            try: df = pd.read_csv(file, encoding='utf-8')
            except: df = pd.read_csv(file, encoding='cp932')
        
        # 項目名自動調整
        header_found = False
        for i in range(min(len(df), 20)):
            row_vals = [str(x) for x in df.iloc[i].values]
            if any(re.search(r'^[RrＲｒ]$|^場所$|^馬名$', str(x).strip()) for x in row_vals):
                df.columns = df.iloc[i]; df = df.iloc[i+1:].reset_index(drop=True); header_found = True; break

        df.columns = df.columns.astype(str).str.strip()
        name_map = {
            '場所': '場名', '開催': '場名', '競馬場': '場名',
            '調教師': '厩舎', '調教師名': '厩舎', '厩舎名': '厩舎',
            '騎手名': '騎手', 'レース': 'R', 'Ｒ': 'R', '番': '正番', '馬番': '正番',
            '単オッズ': '単ｵｯｽﾞ', '単勝オッズ': '単ｵｯｽﾞ', 'オッズ': '単ｵｯｽﾞ', '着': '着順'
        }
        df = df.rename(columns=name_map)
        ensure_cols = ['R', '場名', '馬名', '正番', '騎手', '厩舎', '馬主', '単ｵｯｽﾞ', '着順']
        for col in ensure_cols:
            if col not in df.columns: df[col] = np.nan

        df['R'] = pd.to_numeric(df['R'].apply(to_half_width), errors='coerce')
        df['正番'] = pd.to_numeric(df['正番'].apply(to_half_width), errors='coerce')
        df = df.dropna(subset=['R', '正番'])
        df['R'] = df['R'].astype(int); df['正番'] = df['正番'].astype(int)
        for col in ['騎手', '厩舎', '馬主', '馬名', '場名']:
            df[col] = df[col].apply(normalize_name)
        df['単ｵｯｽﾞ'] = pd.to_numeric(df['単ｵｯｽﾞ'].apply(to_half_width), errors='coerce')
        return df.copy(), "success"
    except Exception as e: return pd.DataFrame(), str(e)

# --- 3. 配置計算エンジン（フラグ管理・AI予測対応） ---
def analyze_haichi(df_curr, df_prev=None):
    df = df_curr.copy()
    if 'タイプ' in df.columns and df['タイプ'].notna().any(): return df

    df['頭数'] = df.groupby(['場名', 'R'])['正番'].transform('max').fillna(16).astype(int)
    df['逆番'] = (df['頭数'] + 1) - df['正番']
    df['正循環'] = df['頭数'] + df['正番']
    df['逆循環'] = df['頭数'] + df['逆番']

    # AI学習・予測用フラグの初期化
    df['青塗フラグ'] = 0
    df['ペアフラグ'] = 0
    df['前日配置フラグ'] = 0

    df['タイプ_list'] = [[] for _ in range(len(df))]
    df['属性_list'] = [[] for _ in range(len(df))]
    df['パターン_list'] = [[] for _ in range(len(df))]
    df['スコア'] = 0.0
    idx_map = {(row['場名'], row['R'], row['正番']): idx for idx, row in df.iterrows()}

    # A. 青塗判定
    for col in ['騎手', '厩舎', '馬主']:
        g_keys = ['場名', col] if col == '騎手' else [col]
        for name, group in df.groupby(g_keys):
            if len(group) < 2 or not name: continue
            all_sets = [{r['正番'], r['逆番'], r['正循環'], r['逆循環']} for _, r in group.iterrows()]
            common = set.intersection(*all_sets)
            if common:
                for _, row in group.iterrows():
                    idx = idx_map.get((row['場名'], row['R'], row['正番']))
                    if idx is not None:
                        df.at[idx, '青塗フラグ'] = 1
                        df.at[idx, 'タイプ_list'].append(f'★{col}青塗')
                        df.at[idx, '属性_list'].append(f'{col}:{name}')
                        df.at[idx, 'パターン_list'].append('青塗')
                        df.at[idx, 'スコア'] += 9.0 + (1.0 if col == '騎手' else 0.2)

    # B. ペア判定
    pair_labels = list("ABCDEFGHIJKLMNOP")
    for col in ['騎手', '厩舎', '馬主']:
        for name, group in df.groupby(['場名', col] if col=='騎手' else col):
            if len(group) < 2 or not name: continue
            rows = group.sort_values('R').to_dict('records')
            for i in range(len(rows)-1):
                r1, r2 = rows[i], rows[i+1]
                v1, v2 = [r1[c] for c in ['正番','逆番','正循環','逆循環']], [r2[c] for c in ['正番','逆番','正循環','逆循環']]
                pats = [pair_labels[x*4+y] for x in range(4) for y in range(4) if v1[x]==v2[y] and v1[x]!=0]
                if pats:
                    df.at[idx_map[(r1['場名'],r1['R'],r1['正番'])], 'ペアフラグ'] = 1
                    df.at[idx_map[(r2['場名'],r2['R'],r2['正番'])], 'ペアフラグ'] = 1
                    is_c = any(x in pats for x in ['C','D','G','H'])
                    for r_data in [r1, r2]:
                        idx = idx_map.get((r_data['場名'], r_data['R'], r_data['正番']))
                        if idx is not None:
                            df.at[idx, 'タイプ_list'].append('◎チャンス' if is_c else '○狙い目')
                            df.at[idx, '属性_list'].append(f'{col}:{name}')
                            df.at[idx, 'パターン_list'].append("".join(pats))
                            df.at[idx, 'スコア'] += 4.0 if is_c else 3.0

    # C. 前日同配置
    if df_prev is not None and not df_prev.empty:
        for idx, row in df.iterrows():
            prev_match = df_prev[(df_prev['場名'] == row['場名']) & (df_prev['R'] == row['R']) & (df_prev['騎手'] == row['騎手'])]
            for _, p_row in prev_match.iterrows():
                if {row['正番'],row['逆番'],row['正循環'],row['逆循環']}.intersection({p_row['正番'],p_row['逆番'],p_row['正循環'],p_row['逆循環']}):
                    df.at[idx, '前日配置フラグ'] = 1
                    df.at[idx, 'タイプ_list'].append('★前日同配置'); df.at[idx, '属性_list'].append(f'前日:騎手:{row["騎手"]}'); df.at[idx, 'パターン_list'].append('前日'); df.at[idx, 'スコア'] += 8.3

    # D. AIによる予測 (モデルが存在する場合)
    if model:
        try:
            X = df[['単ｵｯｽﾞ', '青塗フラグ', 'ペアフラグ', '前日配置フラグ']].fillna(0)
            probs = model.predict_proba(X)
            df['AI激走確率'] = [round(p[1] * 100, 1) for p in probs]
        except:
            df['AI激走確率'] = 0.0

    df['タイプ'] = df['タイプ_list'].apply(lambda x: ' / '.join(x) if isinstance(x, list) else x)
    df['属性'] = df['属性_list'].apply(lambda x: ' / '.join(list(set(x))) if isinstance(x, list) else x)
    df['パターン'] = df['パターン_list'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
    return df

# --- 4. 判定・ランキングロジック ---
def apply_ranking_logic(df_in):
    if df_in.empty: return df_in
    df = df_in.copy()
    df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
    hit_results = df[df['着順'] <= 3]
    hit_attrs = set([a.replace('隣:', '').replace('前日:', '') for _, row in hit_results.iterrows() for a in str(row.get('属性', '')).split(' / ')])
    hit_pats = set([p for pats in hit_results['パターン'].dropna() for p in str(pats).split(',') if p])

    def get_metrics(row):
        score = row.get('スコア', 0); p_list = str(row.get('パターン', '')).split(',')
        ai_prob = row.get('AI激走確率', 0)
        bonus = 4.0 if any(p in hit_pats and len(p)==1 for p in p_list) else 0.0
        
        reasons = []
        for ra in str(row.get('属性', '')).split(' / '):
            cra = ra.replace('隣:', '').replace('前日:', '')
            if cra in hit_attrs: reasons.append(f"{cra.split(':')[0] if ':' in cra else '本人'}好走")
        
        penalty = -3.0 if reasons else 0.0
        # 総合スコアにAIの期待値を少し反映(重み付け)
        total = score + bonus + penalty + (ai_prob / 10.0) + (-30.0 if pd.to_numeric(row.get('単ｵｯｽﾞ'), errors='coerce') > 49.9 else 0.0)
        
        rec = "👑 盤石の軸" if total >= 18 else "✨ 推奨軸" if total >= 14 else "🔥 激走候補" if total >= 11 else "▲ 配置注目" if score > 0 else ""
        return pd.Series([total, f"⚠️{','.join(set(reasons))}(-3)" if reasons else "", rec])

    df[['総合スコア', 'エネルギー状態', '推奨買い目']] = df.apply(get_metrics, axis=1)
    return df

# --- 5. UI（統合UI） ---
st.title("🏇 AI配置馬券 分析・予測統合システム")


with st.sidebar:
    st.header("📂 読み込み")
    up_curr = st.file_uploader("当日データ(Excel/CSV)", type=['xlsx', 'csv'], key="curr")
    up_prev = st.file_uploader("前日データ", type=['xlsx', 'csv'], key="prev")
    
    if up_curr:
        pure_name = up_curr.name.replace('progress_', '').replace('.csv', '').replace('.xlsx', '')
        if "current_pure_name" not in st.session_state or st.session_state["current_pure_name"] != pure_name:
            st.session_state["current_pure_name"] = pure_name
            if "analyzed_df" in st.session_state: del st.session_state["analyzed_df"]; st.rerun()

    if 'analyzed_df' in st.session_state:
        st.divider()
        st.header("💾 経過を保存")
        csv = st.session_state['analyzed_df'].to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 現在の分析(AI・着順込)をCSVで保存", csv, f"progress_{up_curr.name if up_curr else 'data'}.csv")
        if model: st.success("🤖 AIモデル適用中")

if up_curr:
    df_raw, status = load_data(up_curr)
    df_p_raw, _ = load_data(up_prev) if up_prev else (None, None)
    
    if status == "success":
        if 'analyzed_df' not in st.session_state:
            st.session_state['analyzed_df'] = apply_ranking_logic(analyze_haichi(df_raw, df_p_raw))
        
        full_df = st.session_state['analyzed_df']

        # ① 結果入力とAI予測の確認
        st.subheader("📝 結果入力 & AI予測プレビュー")
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
                            # AI確率を表示列に加える
                            disp_cols = ['正番','馬名','単ｵｯｽﾞ','AI激走確率','エネルギー状態','総合スコア','着順','推奨買い目']
                            disp = race_full[race_full['スコア'] > 0].copy()
                            if disp.empty: 
                                st.caption("配置該当なし"); edited_dfs.append(race_full)
                            else:
                                ed = st.data_editor(disp[disp_cols], hide_index=True, use_container_width=True, key=f"ed_{place}_{r_num}", 
                                                    column_config={"AI激走確率": st.column_config.ProgressColumn("AI激走確率", format="%.1f%%", min_value=0, max_value=100)})
                                updated = race_full.copy()
                                for _, row in ed.iterrows(): updated.loc[updated['正番'] == row['正番'], '着順'] = row['着順']
                                edited_dfs.append(updated)
            if st.form_submit_button("🔄 入力を確定して全体を更新"):
                st.session_state['analyzed_df'] = apply_ranking_logic(pd.concat(edited_dfs, ignore_index=True)); st.rerun()

        # ② AI推奨馬（未来レース）
        st.divider(); st.subheader("👑 AI・配置特選馬")
        future_df = full_df[(full_df['着順'].isna()) & (full_df['総合スコア'] >= 11)]
        if not future_df.empty:
            f_tabs = st.tabs(sorted(future_df['場名'].unique()))
            for f_tab, pl in zip(f_tabs, sorted(future_df['場名'].unique())):
                with f_tab:
                    p_f = future_df[future_df['場名'] == pl]
                    fr_tabs = st.tabs([f"{r}R" for r in sorted(p_f['R'].unique())])
                    for fr_tab, r in zip(fr_tabs, sorted(p_f['R'].unique())):
                        with fr_tab: 
                            st.dataframe(p_f[p_f['R'] == r].sort_values('総合スコア', ascending=False)[['正番','馬名','単ｵｯｽﾞ','AI激走確率','エネルギー状態','総合スコア','推奨買い目']], 
                                         column_config={"AI激走確率": st.column_config.ProgressColumn("AI激走確率", format="%.1f%%", min_value=0, max_value=100)},
                                         use_container_width=True, hide_index=True)

        # ③ 統計分析
        st.divider(); st.subheader("📈 的中統計")
        df_res = full_df[full_df['着順'].notna()].copy()
        if not df_res.empty:
            s_tabs = st.tabs(["合計"] + sorted(full_df['場名'].unique()))
            for s_tab, s_pl in zip(s_tabs, ["合計"] + sorted(full_df['場名'].unique())):
                with s_tab:
                    df_s = df_res if s_pl == "合計" else df_res[df_res['場名'] == s_pl]
                    df_fk = df_s[df_s['着順'] <= 3]
                    if df_s.empty: st.info("データなし")
                    else:
                        c_m, c_c = st.columns([1, 2])
                        with c_m: 
                            st.metric("消化頭数", len(df_s)); st.metric("複勝率", f"{len(df_fk)/len(df_s)*100 if len(df_s)>0 else 0:.1f}%"); st.metric("AI平均確率", f"{df_fk['AI激走確率'].mean():.1f}%")
                        with c_c:
                            all_p = [p for pats in df_fk['パターン'] for p in str(pats).split(',') if p]
                            if all_p: st.plotly_chart(px.pie(pd.Series(all_p).value_counts().reset_index(), values='count', names='index', title=f'【{s_pl}】的中パターン', hole=0.4), use_container_width=True)
