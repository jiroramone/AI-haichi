import streamlit as st
import pandas as pd
import numpy as np
import re

# --- 1. 基本設定 & ユーティリティ ---
st.set_page_config(page_title="配置馬券術AI分析システム", layout="wide")

# ポイント配分設定
HAICHI_POINTS = {
    'pair_jockey': 3.0,          # 騎手ペア
    'pair_stable_owner': 1.0,    # 厩舎・馬主ペア
    'blue_jockey': 4.0,          # 騎手青塗
    'blue_stable_owner': 2.0,    # 厩舎・馬主青塗
    'blue_neighbor': 2.0,        # 青塗隣
    'sandwich_bonus': 4.0,       # 青塗サンドイッチ
    'stable_symmetry': 2.0,      # 厩舎対称配置
    'stable_symmetry_neighbor': 2.0, # 厩舎対称配置(隣)
    'continuous': 1.0,           # 連続レース配置
    'odds_rank_bonus': 1.0,      # 1~5番人気
    'prev_day_same_fail': 1.0,   # 前日同R同配置で凡走
    'prev_day_same_win': -1.0,   # 前日同R同配置で好走
    'trend_bonus': 2.0,          # トレンド加算
}

def to_half_width(text):
    if pd.isna(text): return text
    text = str(text)
    table = str.maketrans('０１２３４５６７８９．', '0123456789.')
    return re.sub(r'[^\d\.]', '', text.translate(table))

def normalize_name(x):
    if pd.isna(x): return ''
    s = str(x).strip()
    s = s.replace('　', '').replace(' ', '')
    return re.sub(r'[★☆▲△◇$*]', '', s)

# --- 2. データ読み込み ---
@st.cache_data
def load_data(file):
    try:
        # エクセルまたはCSVの読み込み
        if file.name.endswith('.xlsx'):
            xls = pd.ExcelFile(file, engine='openpyxl')
            sheet_names = xls.sheet_names
            target_sheet = sheet_names[0]
            for sheet in sheet_names:
                if "全出走馬" in sheet or "出走" in sheet:
                    target_sheet = sheet
                    break
            df = pd.read_excel(xls, sheet_name=target_sheet)
        else:
            try: df = pd.read_csv(file, encoding='utf-8')
            except: df = pd.read_csv(file, encoding='cp932')
        
        # ヘッダー行の自動探索
        if not any(col in str(df.columns) for col in ['馬', '番', 'R', '騎']):
            for i in range(min(len(df), 10)):
                if any(x in str(df.iloc[i].values) for x in ['馬', '番', 'R']):
                    df.columns = df.iloc[i]
                    df = df.iloc[i+1:].reset_index(drop=True)
                    break

        df.columns = df.columns.astype(str).str.strip()
        
        # カラム名の統一
        name_map = {
            '場所': '場名', '開催': '場名', '競馬場': '場名',
            '調教師': '厩舎', '調教師名': '厩舎', '厩舎名': '厩舎',
            '騎手名': '騎手', 'レース': 'R', 'Ｒ': 'R', '番': '正番', '馬番': '正番',
            '単オッズ': '単ｵｯｽﾞ', '単勝オッズ': '単ｵｯｽﾞ', 'オッズ': '単ｵｯｽﾞ',
            '着': '着順', '着順': '着順'
        }
        df = df.rename(columns=name_map)
        
        # 必須カラムの確保
        ensure_cols = ['場名', 'R', '馬名', '正番', '騎手', '厩舎', '馬主', '単ｵｯｽﾞ', '着順']
        for col in ensure_cols:
            if col not in df.columns:
                df[col] = np.nan

        # 数値変換とクリーニング
        df['R'] = pd.to_numeric(df['R'].apply(to_half_width), errors='coerce')
        df['正番'] = pd.to_numeric(df['正番'].apply(to_half_width), errors='coerce')
        df = df.dropna(subset=['R', '正番'])
        df['R'] = df['R'].astype(int)
        df['正番'] = df['正番'].astype(int)
        
        for col in ['騎手', '厩舎', '馬主', '馬名', '場名']:
            df[col] = df[col].apply(normalize_name)
            
        df['単ｵｯｽﾞ'] = pd.to_numeric(df['単ｵｯｽﾞ'].apply(to_half_width), errors='coerce')

        # 着順のクリーニング
        df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
        df.loc[(df['着順'] <= 0) | (df['着順'] > 18), '着順'] = np.nan
        
        return df.copy(), "success"
    except Exception as e: return pd.DataFrame(), str(e)

# --- 3. 分析エンジン ---
def identify_pair_patterns(r1, r2):
    patterns = []
    s1, s2 = r1['正番'], r2['正番']
    g1, g2 = r1['逆番'], r2['逆番']
    sj1, sj2 = r1['正循環'], r2['正循環']
    gj1, gj2 = r1['逆循環'], r2['逆循環']
    
    if s1 == s2: patterns.append('A')
    if s1 == g2: patterns.append('B')
    if s1 == sj2: patterns.append('C')
    if s1 == gj2: patterns.append('D')
    if g1 == s2: patterns.append('E')
    if g1 == g2: patterns.append('F')
    if g1 == sj2: patterns.append('G')
    if g1 == gj2: patterns.append('H')
    if sj1 == s2: patterns.append('I')
    if sj1 == g2: patterns.append('J')
    if sj1 == sj2: patterns.append('K')
    if sj1 == gj2: patterns.append('L')
    if gj1 == s2: patterns.append('M')
    if gj1 == g2: patterns.append('N')
    if gj1 == sj2: patterns.append('O')
    if gj1 == gj2: patterns.append('P')
    if s1 != s2 and (s1 % 10 == s2 % 10):
        if s1 < s2: patterns.append('Q')
        else: patterns.append('R')
    return patterns

def extract_patterns(row):
    found_patterns = set()
    for val in row.values:
        s_val = str(val).strip()
        if re.match(r'^[A-R]$', s_val):
            found_patterns.add(s_val)
    return list(found_patterns)

def analyze_haichi_advanced(df_curr, df_prev=None):
    df = df_curr.copy()
    
    # 強制クリーニング
    df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
    df.loc[(df['着順'] <= 0) | (df['着順'] > 18), '着順'] = np.nan
    
    max_umaban = df.groupby(['場名', 'R'])['正番'].transform('max')
    df['頭数'] = max_umaban.fillna(16).astype(int)
    df['逆番'] = (df['頭数'] + 1) - df['正番']
    df['正循環'] = df['頭数'] + df['正番']
    df['逆循環'] = df['頭数'] + df['逆番']
    
    for c in ['正番', '逆番', '正循環', '逆循環']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

    df['パターンリスト'] = df.apply(extract_patterns, axis=1)

    if '合計ポイント' not in df.columns: df['合計ポイント'] = 0.0
    if '動的ポイント' not in df.columns: df['動的ポイント'] = 0.0
    if 'トレンドポイント' not in df.columns: df['トレンドポイント'] = 0.0
    
    if '属性_list' not in df.columns:
        df['属性_list'] = [[] for _ in range(len(df))]
    else:
        df['合計ポイント'] = 0.0 
        df['属性_list'] = [[] for _ in range(len(df))]
        
    df['ペア対象_list'] = [[] for _ in range(len(df))] 

    idx_map = {(row['場名'], row['R'], row['正番']): i for i, row in df.iterrows()}

    # --- A. 青塗 ---
    blue_paint_targets = []
    for category in ['騎手', '厩舎', '馬主']:
        for (place, name), group in df.groupby(['場名', category]):
            if len(group) < 2: continue
            
            group['正循環'] = group['頭数'] + group['正番']
            group['逆循環'] = group['頭数'] + group['逆番']
            
            sets_list = [{r['正番'], r['逆番'], r['正循環'], r['逆循環']} for _, r in group.iterrows()]
            common_nums = set.intersection(*sets_list)
            
            if common_nums:
                num_str = list(common_nums)[0]
                pt = HAICHI_POINTS['blue_jockey'] if category == '騎手' else HAICHI_POINTS['blue_stable_owner']
                for _, row in group.iterrows():
                    idx = idx_map.get((row['場名'], row['R'], row['正番']))
                    if idx is not None:
                        df.at[idx, '合計ポイント'] += pt
                        df.at[idx, '属性_list'].append(f"★{category}青塗(No.{num_str})")
                        blue_paint_targets.append({'場名': place, 'R': row['R'], '正番': row['正番'], 'cat': category})

    # --- B. 青塗隣 & サンドイッチ ---
    blue_map = {}
    for b in blue_paint_targets:
        key = (b['場名'], b['R'], b['正番'])
        if key not in blue_map: blue_map[key] = []
        blue_map[key].append(b['cat'])

    for b in blue_paint_targets:
        for neighbor_num in [b['正番'] - 1, b['正番'] + 1]:
            idx = idx_map.get((b['場名'], b['R'], neighbor_num))
            if idx is not None:
                current_attrs = df.at[idx, '属性_list']
                tag = f"△{b['cat']}青塗隣"
                if tag not in current_attrs:
                    df.at[idx, '合計ポイント'] += HAICHI_POINTS['blue_neighbor']
                    df.at[idx, '属性_list'].append(tag)

    for idx, row in df.iterrows():
        my_num = row['正番']
        left_key = (row['場名'], row['R'], my_num - 1)
        right_key = (row['場名'], row['R'], my_num + 1)
        if left_key in blue_map and right_key in blue_map:
            my_odds = pd.to_numeric(row['単ｵｯｽﾞ'], errors='coerce')
            l_row = idx_map.get(left_key)
            r_row = idx_map.get(right_key)
            l_odds = pd.to_numeric(df.at[l_row, '単ｵｯｽﾞ'], errors='coerce') if l_row else np.nan
            r_odds = pd.to_numeric(df.at[r_row, '単ｵｯｽﾞ'], errors='coerce') if r_row else np.nan
            
            if pd.notna(my_odds) and pd.notna(l_odds) and pd.notna(r_odds):
                if my_odds < l_odds and my_odds < r_odds:
                    df.at[idx, '合計ポイント'] += HAICHI_POINTS['sandwich_bonus']
                    df.at[idx, '属性_list'].append("🔥青塗サンドイッチ(好配置)")

    # --- C. ペア ---
    for category in ['騎手', '厩舎', '馬主']:
        for (place, name), group in df.groupby(['場名', category]):
            if len(group) < 2: continue
            
            group['正循環'] = group['頭数'] + group['正番']
            group['逆循環'] = group['頭数'] + group['逆番']
            rows = group.sort_values('R').to_dict('records')
            
            pt_pair = HAICHI_POINTS['pair_jockey'] if category == '騎手' else HAICHI_POINTS['pair_stable_owner']
            
            for i in range(len(rows) - 1):
                r1, r2 = rows[i], rows[i+1]
                patterns = identify_pair_patterns(r1, r2)
                
                if patterns:
                    is_continuous = (r2['R'] - r1['R'] == 1)
                    bonus = HAICHI_POINTS['continuous'] if is_continuous else 0
                    pattern_str = ",".join(patterns)
                    
                    for r_data in [r1, r2]:
                        idx = idx_map.get((r_data['場名'], r_data['R'], r_data['正番']))
                        if idx is not None:
                            tag = f"○{category}ペア({pattern_str})" + ("(連続)" if is_continuous else "")
                            current_list = df.at[idx, '属性_list']
                            if not any(f"○{category}ペア" in x for x in current_list):
                                df.at[idx, '合計ポイント'] += pt_pair + bonus
                                df.at[idx, '属性_list'].append(tag)
                            target_r = r2['R'] if r_data['R'] == r1['R'] else r1['R']
                            df.at[idx, 'ペア対象_list'].append({'R': target_r, 'cat': category})

    # --- D. 対称 & 対称隣 ---
    for (place, r), race_group in df.groupby(['場名', 'R']):
        symmetry_targets = []
        for stable_name, stable_group in race_group.groupby('厩舎'):
            if len(stable_group) < 2: continue
            stable_group['正循環'] = stable_group['頭数'] + stable_group['正番']
            stable_group['逆循環'] = stable_group['頭数'] + stable_group['逆番']
            s_rows = stable_group.to_dict('records')
            has_symmetry = False
            for i in range(len(s_rows)):
                for j in range(i + 1, len(s_rows)):
                    s1 = {s_rows[i]['正番'], s_rows[i]['逆番'], s_rows[i]['正循環'], s_rows[i]['逆循環']}
                    s2 = {s_rows[j]['正番'], s_rows[j]['逆番'], s_rows[j]['正循環'], s_rows[j]['逆循環']}
                    if s1.intersection(s2): has_symmetry = True
            if has_symmetry:
                for idx_s, row_s in stable_group.iterrows():
                    symmetry_targets.append(row_s['正番'])
                    if "◇厩舎対称" not in df.at[idx_s, '属性_list']:
                        df.at[idx_s, '合計ポイント'] += HAICHI_POINTS['stable_symmetry']
                        df.at[idx_s, '属性_list'].append("◇厩舎対称")

        for sym_num in symmetry_targets:
            for neighbor_num in [sym_num - 1, sym_num + 1]:
                idx = idx_map.get((place, r, neighbor_num))
                if idx is not None:
                    tag = "◆厩舎対称隣"
                    if tag not in df.at[idx, '属性_list']:
                        df.at[idx, '合計ポイント'] += HAICHI_POINTS['stable_symmetry_neighbor']
                        df.at[idx, '属性_list'].append(tag)

    # --- E. 前日比較 ---
    if df_prev is not None and not df_prev.empty:
        prev_map = {}
        for _, row in df_prev.iterrows():
            key = (row['R'], row['正番']) 
            prev_map[key] = pd.to_numeric(row['着順'], errors='coerce')
        for idx, row in df.iterrows():
            key = (row['R'], row['正番'])
            if key in prev_map:
                prev_rank = prev_map[key]
                if pd.notna(prev_rank):
                    if prev_rank > 3:
                        df.at[idx, '合計ポイント'] += HAICHI_POINTS['prev_day_same_fail']
                        df.at[idx, '属性_list'].append("★前日同配置(凡走)")
                    elif prev_rank <= 3:
                        df.at[idx, '合計ポイント'] += HAICHI_POINTS['prev_day_same_win']
                        df.at[idx, '属性_list'].append("▼前日同配置(好走)")

    # --- F. 人気加点 ---
    if '単ｵｯｽﾞ' in df.columns:
        df['人気ランク'] = df.groupby(['場名', 'R'])['単ｵｯｽﾞ'].rank(method='min')
        df.loc[df['人気ランク'] <= 5, '合計ポイント'] += HAICHI_POINTS['odds_rank_bonus']

    df['属性'] = df['属性_list'].apply(lambda x: ' / '.join(x))
    return df

# --- 4. 動的ロジック & トレンド分析 ---
def calculate_place_trends(df):
    trends = {} 
    finished = df[pd.notna(pd.to_numeric(df['着順'], errors='coerce'))].copy()
    if finished.empty: return trends
    finished['is_win'] = pd.to_numeric(finished['着順'], errors='coerce') <= 3
    
    for place, place_df in finished.groupby('場名'):
        trends[place] = {}
        attr_stats = {}
        for _, row in place_df.iterrows():
            attrs = str(row['属性']).split(' / ')
            for attr in attrs:
                if 'トレンド' in attr: continue
                match = re.search(r'ペア\(([A-R])\)', attr)
                if match: key = f"パターン{match.group(1)}"
                elif "青塗" in attr:
                    if "騎手" in attr: key = "騎手青塗"
                    elif "厩舎" in attr: key = "厩舎青塗"
                    elif "隣" in attr: key = "青塗隣"
                    else: key = "青塗(他)"
                elif "対称" in attr:
                    if "隣" in attr: key = "厩舎対称隣"
                    else: key = "厩舎対称"
                else: continue
                if key not in attr_stats: attr_stats[key] = {'wins': 0, 'total': 0}
                attr_stats[key]['total'] += 1
                if row['is_win']: attr_stats[key]['wins'] += 1
        
        for key, stat in attr_stats.items():
            if stat['total'] >= 2:
                rate = stat['wins'] / stat['total']
                if rate >= 0.4:
                    trends[place][key] = {'rate': rate, 'count': stat['total']}
    return trends

def update_dynamic_points_chain(df):
    if '着順' not in df.columns: return df
    
    df['動的ポイント'] = 0.0
    bonus_map = {} 
    for category in ['騎手', '厩舎', '馬主']:
        for (place, name), group in df.groupby(['場名', category]):
            if len(group) < 2: continue
            rows = group.sort_values('R').to_dict('records')
            expectation_alive = True 
            for i in range(len(rows)):
                curr_row = rows[i]
                mask = (df['場名'] == curr_row['場名']) & (df['R'] == curr_row['R']) & (df['正番'] == curr_row['正番'])
                if mask.sum() == 0: continue
                curr_idx = df[mask].index[0]
                rank = pd.to_numeric(curr_row['着順'], errors='coerce')
                is_finished = pd.notna(rank)
                is_win = is_finished and rank <= 3
                if not expectation_alive:
                    bonus_map[curr_idx] = bonus_map.get(curr_idx, 0) - 2.0
                elif not is_finished:
                    if i > 0: bonus_map[curr_idx] = bonus_map.get(curr_idx, 0) + 2.0
                if is_win: expectation_alive = False
    for idx, bonus in bonus_map.items():
        df.at[idx, '動的ポイント'] += bonus
        df.at[idx, '合計ポイント'] += bonus 

    df['トレンドポイント'] = 0.0
    trends = calculate_place_trends(df)
    st.session_state['current_trends'] = trends
    
    future_mask = pd.isna(pd.to_numeric(df['着順'], errors='coerce'))
    
    for idx in df[future_mask].index:
        place = df.at[idx, '場名']
        attrs = str(df.at[idx, '属性'])
        p_list = df.at[idx, 'パターンリスト']
        if place in trends:
            for trend_key, data in trends[place].items():
                is_match = False
                if trend_key.startswith("パターン"):
                    p_char = trend_key.replace("パターン", "")
                    if f"({p_char})" in attrs: is_match = True
                elif trend_key == "騎手青塗" and "騎手青塗" in attrs: is_match = True
                elif trend_key == "厩舎青塗" and "厩舎青塗" in attrs: is_match = True
                elif trend_key == "青塗隣" and "青塗隣" in attrs: is_match = True
                elif trend_key == "厩舎対称" and "厩舎対称" in attrs and "隣" not in attrs: is_match = True
                elif trend_key == "厩舎対称隣" and "厩舎対称隣" in attrs: is_match = True
                
                if is_match:
                    if f"📈{trend_key}" not in str(df.at[idx, '属性']):
                        bonus = HAICHI_POINTS['trend_bonus']
                        df.at[idx, 'トレンドポイント'] += bonus
                        df.at[idx, '合計ポイント'] += bonus
                        df.at[idx, '属性'] = f"📈傾向({trend_key}) / " + df.at[idx, '属性']
    return df

# --- 5. UIコンポーネント ---
def render_trend_sidebar():
    if 'current_trends' in st.session_state and st.session_state['current_trends']:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 今日の会場別トレンド")
        trends = st.session_state['current_trends']
        for place, data in trends.items():
            if not data: continue
            st.sidebar.markdown(f"**{place}**")
            for kw, val in data.items():
                rate_pct = int(val['rate'] * 100)
                if "パターン" in kw:
                    st.sidebar.markdown(f"- :red[{kw}]: 好走率**{rate_pct}%** ({val['count']}件)")
                else:
                    st.sidebar.caption(f"- {kw}: 好走率**{rate_pct}%** ({val['count']}件)")
            st.sidebar.markdown("---")

def render_race_forecast(full_df):
    st.markdown("### 🎯 厳選勝負レース (推奨買い目)")
    df = full_df.copy()
    
    future_mask = pd.to_numeric(df['着順'], errors='coerce').isna()
    if not future_mask.any():
        st.info("全てのレースが終了しています（または着順が入力済みです）。")
        return

    def check_blue_reverse(row, context_df):
        my_attrs = str(row.get('属性', ''))
        if not my_attrs or '△' not in my_attrs or '青塗隣' not in my_attrs:
            return False, False, ""
        is_jockey_neighbor = '△騎手青塗隣' in my_attrs
        my_num = row['正番']
        my_odds = pd.to_numeric(row['単ｵｯｽﾞ'], errors='coerce')
        if pd.isna(my_odds): return False, False, ""
        race_df = context_df[(context_df['場名'] == row['場名']) & (context_df['R'] == row['R'])]
        for offset in [-1, 1]:
            neighbor_num = my_num + offset
            n_row = race_df[race_df['正番'] == neighbor_num]
            if n_row.empty: continue
            n_attrs = str(n_row.iloc[0].get('属性', ''))
            if '★' in n_attrs and '青塗' in n_attrs:
                n_odds = pd.to_numeric(n_row.iloc[0]['単ｵｯｽﾞ'], errors='coerce')
                if pd.notna(n_odds) and my_odds < n_odds:
                    neighbor_name = n_row.iloc[0]['馬名']
                    return True, is_jockey_neighbor, f"🔥隣の{neighbor_name}(青塗)より人気"
        return False, False, ""

    def calculate_rank_and_reason(row, context_df):
        odds = pd.to_numeric(row['単ｵｯｽﾞ'], errors='coerce')
        if pd.isna(odds) or odds > 49.9: return "C", row['属性']
        if row.get('動的ポイント', 0) < 0: return "C", row['属性']
        
        is_reverse, is_jockey_origin, reverse_msg = check_blue_reverse(row, context_df)
        new_reason = row['属性']
        if is_reverse:
            new_reason = f"【鉄板】{reverse_msg} / " + new_reason
            if is_jockey_origin: return "SS", new_reason
            else: return "S", new_reason
        if row.get('動的ポイント', 0) > 0:
            return "S", f"【激熱】直前ペア凡走 / {new_reason}"
        if row.get('合計ポイント', 0) >= 10.0: return "A", new_reason
        if row.get('合計ポイント', 0) >= 7.0: return "B", new_reason
        return "C", new_reason

    places = sorted(df['場名'].unique())
    has_recommendation = False
    p_tabs = st.tabs(places)
    
    for p_tab, place in zip(p_tabs, places):
        with p_tab:
            place_df = df[df['場名'] == place]
            target_races = []
            race_data_map = {} 

            all_races = sorted(place_df['R'].unique())
            for r_num in all_races:
                race_df = place_df[place_df['R'] == r_num].copy()
                if race_df[pd.to_numeric(race_df['着順'], errors='coerce').isna()].empty:
                    continue
                
                results = race_df.apply(lambda x: calculate_rank_and_reason(x, df), axis=1)
                race_df['ランク'] = [r[0] for r in results]
                race_df['拡張根拠'] = [r[1] for r in results]
                
                axis_candidates = race_df[race_df['ランク'].isin(['SS', 'S'])]
                if not axis_candidates.empty:
                    rank_map = {'SS': 3, 'S': 2}
                    axis_candidates['rank_score'] = axis_candidates['ランク'].map(rank_map)
                    axis_horse = axis_candidates.sort_values(['rank_score', '合計ポイント'], ascending=[False, False]).iloc[0]
                    
                    opponents = race_df[
                        (race_df['正番'] != axis_horse['正番']) &
                        (race_df['動的ポイント'] >= 0) &
                        ((race_df['合計ポイント'] >= 3.0) | (race_df['人気ランク'] <= 5))
                    ].sort_values(['合計ポイント', '人気ランク'], ascending=[False, True]).head(4)
                    
                    opp_nums = opponents['正番'].astype(str).tolist()
                    opp_str = ",".join(opp_nums)
                    
                    target_races.append(r_num)
                    race_data_map[r_num] = (axis_horse, opp_nums, opp_str)
                    has_recommendation = True

            if not target_races:
                st.info("この会場には推奨レースがありません。")
                continue

            r_tabs = st.tabs([f"{r}R" for r in target_races])
            for r_tab, r_num in zip(r_tabs, target_races):
                with r_tab:
                    axis_horse, opp_nums, opp_str = race_data_map[r_num]
                    with st.container():
                        rank_color = "red" if axis_horse['ランク'] == "SS" else "orange"
                        st.markdown(f"##### :{rank_color}[【{axis_horse['ランク']}】 {place} {r_num}R] 軸: {axis_horse['正番']} {axis_horse['馬名']} ({axis_horse['騎手']})")
                        c1, c2 = st.columns([2, 3])
                        with c1:
                            st.info(f"**根拠**: {axis_horse['拡張根拠']}")
                            st.write(f"オッズ: **{axis_horse['単ｵｯｽﾞ']}**倍 / スコア: **{axis_horse['合計ポイント']}**")
                        with c2:
                            st.write("**🎫 推奨買い目**")
                            bets = []
                            an = axis_horse['正番']
                            if opp_nums:
                                bets.append(f"- **ワイド**: {an} － {opp_str}")
                                if axis_horse['ランク'] == 'SS' or float(axis_horse['単ｵｯｽﾞ']) >= 10:
                                    bets.append(f"- **3連複**: {an} － {opp_str} (ボーナス)")
                            else:
                                bets.append("- 単勝推奨 (相手不在)")
                            for b in bets: st.write(b)
                        st.markdown("---")

    if not has_recommendation:
        st.info("現在、厳選条件（SS/Sランク）に合致する勝負レースはありません。")

def render_main_tabs(full_df):
    places = sorted(full_df['場名'].unique())
    if not places: return
    p_tabs = st.tabs(places)
    for p_tab, place in zip(p_tabs, places):
        with p_tab:
            place_df = full_df[full_df['場名'] == place]
            races = sorted(place_df['R'].unique())
            r_tabs = st.tabs([f"{r}R" for r in races])
            for r_tab, r_num in zip(r_tabs, races):
                with r_tab:
                    race_df = place_df[place_df['R'] == r_num].sort_values('正番').copy()
                    with st.expander("📝 結果入力・修正", expanded=True):
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            if '着順' not in race_df.columns: race_df['着順'] = None
                            edited = st.data_editor(
                                race_df[['正番', '馬名', '着順']],
                                column_config={
                                    "正番": st.column_config.NumberColumn(disabled=True, width="small"),
                                    "馬名": st.column_config.TextColumn("馬名", width="medium"),
                                    "着順": st.column_config.NumberColumn("着順", min_value=1, max_value=18, format="%d")
                                },
                                hide_index=True, use_container_width=True, key=f"ed_{place}_{r_num}"
                            )
                        with c2:
                            st.write(""); st.write("")
                            if st.button("更新", key=f"btn_{place}_{r_num}"):
                                updates = edited.set_index('正番')['着順'].to_dict()
                                full_current = st.session_state['analyzed_df']
                                for idx in full_current[(full_current['場名']==place) & (full_current['R']==r_num)].index:
                                    n = full_current.at[idx, '正番']
                                    full_current.at[idx, '着順'] = updates.get(n)
                                new_df = update_dynamic_points_chain(full_current)
                                st.session_state['analyzed_df'] = new_df
                                st.rerun()
                    st.markdown("##### 📊 分析チャート")
                    disp = race_df.copy()
                    def get_status(row):
                        if row['動的ポイント'] > 0: return "🔥激熱"
                        if row['動的ポイント'] < 0: return "🛑終了"
                        if row['合計ポイント'] >= 10: return "⭐本命"
                        return "―"
                    disp['状態'] = disp.apply(get_status, axis=1)
                    def get_link(row):
                        links = []
                        for t in row.get('ペア対象_list', []):
                            icon = "🔙" if t['R'] < row['R'] else "🔜"
                            links.append(f"{icon}{t['R']}R")
                        return " ".join(links)
                    disp['連動'] = disp.apply(get_link, axis=1)
                    st.dataframe(
                        disp[['枠番', '正番', '馬名', '騎手', '単ｵｯｽﾞ', '合計ポイント', '動的ポイント', 'トレンドポイント', '状態', '連動', '属性']],
                        column_config={
                            "馬名": st.column_config.TextColumn("馬名", width="medium"),
                            "合計ポイント": st.column_config.ProgressColumn("スコア", format="%.1f", min_value=-5, max_value=20),
                            "動的ポイント": st.column_config.NumberColumn("補正", format="%+.1f"),
                            "トレンドポイント": st.column_config.NumberColumn("傾向", format="%+.1f"),
                            "状態": st.column_config.TextColumn("判定", width="small"),
                            "属性": st.column_config.TextColumn("根拠", width="large"),
                        },
                        hide_index=True, use_container_width=True
                    )

# --- 7. メイン処理フロー ---
def main():
    st.sidebar.title("🏇 設定・データ")
    st.sidebar.subheader("💾 途中経過の読み込み")
    uploaded_progress = st.sidebar.file_uploader("保存したCSVを読み込む", type=['csv'], key="progress")
    st.sidebar.subheader("🆕 新規分析")
    uploaded_curr = st.sidebar.file_uploader("当日出馬表 (必須)", type=['xlsx', 'csv'], key="curr")
    uploaded_prev = st.sidebar.file_uploader("前日出馬表 (土日連動用)", type=['xlsx', 'csv'], key="prev")
    
    full_df = None # 変数を初期化

    if uploaded_progress:
        try:
            df = pd.read_csv(uploaded_progress)
            st.session_state['analyzed_df'] = df
            st.sidebar.success("復元しました！")
        except Exception as e:
            st.sidebar.error(f"復元エラー: {e}")
    elif uploaded_curr:
        curr_name = uploaded_curr.name
        prev_name = uploaded_prev.name if uploaded_prev else "none"
        session_key = f"{curr_name}_{prev_name}"
        if 'last_session_key' not in st.session_state or st.session_state['last_session_key'] != session_key:
            df_curr, msg1 = load_data(uploaded_curr)
            df_prev, msg2 = (None, "")
            if uploaded_prev: df_prev, msg2 = load_data(uploaded_prev)
            if msg1 == "success":
                with st.spinner("AI分析を実行中..."):
                    df = analyze_haichi_advanced(df_curr, df_prev)
                    df = update_dynamic_points_chain(df)
                st.session_state['analyzed_df'] = df
                st.session_state['last_session_key'] = session_key
            else:
                st.error(f"読み込みエラー: {msg1}")
    
    if 'analyzed_df' in st.session_state:
        full_df = st.session_state['analyzed_df']
        
        # 強制リセット: メモリ内のデータに対しても「0→NaN」を適用
        full_df['着順'] = pd.to_numeric(full_df['着順'], errors='coerce')
        full_df.loc[(full_df['着順'] <= 0) | (full_df['着順'] > 18), '着順'] = np.nan
        st.session_state['analyzed_df'] = full_df # 更新
        
        # サイドバーにリセットボタンを設置
        st.sidebar.markdown("---")
        if st.sidebar.button("⚠️ 着順データを全リセット", help="『終了』と誤判定される場合に押してください。"):
            full_df['着順'] = np.nan
            full_df = update_dynamic_points_chain(full_df)
            st.session_state['analyzed_df'] = full_df
            st.success("着順データを全てリセットしました！")
            st.rerun()

    # full_dfが有効な場合のみ描画処理を行う
    if full_df is not None:
        render_trend_sidebar()
        csv = full_df.to_csv(index=False).encode('utf-8-sig')
        st.sidebar.download_button(
            "💾 分析データを保存 (CSV)",
            csv,
            "haichi_analysis_save.csv",
            "text/csv",
            help="現在の状態を保存"
        )
        render_race_forecast(full_df)
        render_main_tabs(full_df)
    else:
        st.info("👈 サイドバーからデータをアップロードしてください。\n(例: 260104全出走馬.csv)")

if __name__ == "__main__":
    main()
