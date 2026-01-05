import streamlit as st
import pandas as pd
import numpy as np
import re
import itertools
import os

# --- 1. 基本設定 & ユーティリティ ---
st.set_page_config(page_title="配置馬券術AI分析システム", layout="wide")

LEARNING_FILE = "haichi_learning_data.csv"

# --- 2025年JRAリーディング確定版データ ---
JOCKEYS_SS = [
    "ルメール", "戸崎圭太", "松山弘平", "横山武史", "坂井瑠星", "川田将雅"
]
JOCKEYS_KANTO = [
    "丹内祐次", "佐々木大輔", "横山和生", "菅原明良", 
    "三浦皇成", "津村明秀", "横山典弘", "田辺裕信"
]
JOCKEYS_KANSAI = [
    "岩田望来", "高杉吏麒", "北村友一", "武豊", 
    "団野大成", "鮫島克駿", "吉村誠之助", "荻野極", "西村淳也"
]

PLACE_KANTO = ['中山', '東京', '福島', '新潟']
PLACE_KANSAI = ['京都', '阪神', '中京', '小倉', '札幌', '函館'] 

# ポイント配分設定
DEFAULT_POINTS = {
    'pair_jockey': 4.0,          
    'pair_stable_owner': 0.5,    
    'blue_jockey': 5.0,          
    'blue_stable_owner': 1.0,    
    'blue_neighbor': 2.0,        
    'sandwich_bonus': 5.0,       
    'stable_symmetry': 1.0,      
    'stable_symmetry_neighbor': 1.0, 
    'continuous': 2.0,           
    'odds_rank_bonus': 1.0,      
    'prev_day_same_fail': 2.0,   
    'prev_day_same_win': -2.0,   
    'trend_bonus': 3.0,          
    'learning_bonus': 3.0,       
    'leading_jockey_bonus': 2.0  
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
    s = re.split(r'[\(（]', s)[0]
    return re.sub(r'[★☆▲△◇$*]', '', s)

# --- 2. データ読み込み ---
@st.cache_data
def load_data(file):
    try:
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
        
        if not any(col in str(df.columns) for col in ['馬', '番', 'R', '騎']):
            for i in range(min(len(df), 10)):
                if any(x in str(df.iloc[i].values) for x in ['馬', '番', 'R']):
                    df.columns = df.iloc[i]
                    df = df.iloc[i+1:].reset_index(drop=True)
                    break

        df.columns = df.columns.astype(str).str.strip()
        
        name_map = {
            '場所': '場名', '開催': '場名', '競馬場': '場名', '開催会場': '場名',
            '調教師': '厩舎', '調教師名': '厩舎', '厩舎名': '厩舎',
            '騎手名': '騎手', 'レース': 'R', 'Ｒ': 'R', '番': '正番', '馬番': '正番',
            '単オッズ': '単ｵｯｽﾞ', '単勝オッズ': '単ｵｯｽﾞ', 'オッズ': '単ｵｯｽﾞ',
            '着': '着順', '着順': '着順'
        }
        df = df.rename(columns=name_map)
        
        if '着順' in df.columns: df = df.drop(columns=['着順'])
        df['着順'] = np.nan 

        if '厩舎' not in df.columns:
            cols = df.columns.tolist()
            if '斤量' in cols:
                idx_w = cols.index('斤量')
                if idx_w + 1 < len(cols):
                    potential_col = cols[idx_w + 1]
                    df = df.rename(columns={potential_col: '厩舎'})
                if idx_w - 2 >= 0 and '騎手' not in df.columns:
                    potential_jockey = cols[idx_w - 2]
                    df = df.rename(columns={potential_jockey: '騎手'})

        ensure_cols = ['場名', 'R', '馬名', '正番', '騎手', '厩舎', '馬主', '単ｵｯｽﾞ']
        for col in ensure_cols:
            if col not in df.columns: df[col] = np.nan

        # ゴミデータの排除
        if '場名' in df.columns:
            df = df[~df['場名'].astype(str).isin(['場所', '開催', '開催会場', '場名', 'nan'])]

        df['R'] = pd.to_numeric(df['R'].apply(to_half_width), errors='coerce')
        df['正番'] = pd.to_numeric(df['正番'].apply(to_half_width), errors='coerce')
        df = df.dropna(subset=['R', '正番'])
        df['R'] = df['R'].astype(int)
        df['正番'] = df['正番'].astype(int)
        
        for col in ['騎手', '厩舎', '馬主', '馬名', '場名']:
            df[col] = df[col].apply(normalize_name)
            
        df['単ｵｯｽﾞ'] = pd.to_numeric(df['単ｵｯｽﾞ'].apply(to_half_width), errors='coerce')
        
        return df.copy(), "success"
    except Exception as e: return pd.DataFrame(), str(e)

# --- 3. 学習機能 ---
def load_learning_data():
    if os.path.exists(LEARNING_FILE):
        try:
            return pd.read_csv(LEARNING_FILE)
        except:
            return pd.DataFrame(columns=['場名', 'パターン', '着順', '日付'])
    return pd.DataFrame(columns=['場名', 'パターン', '着順', '日付'])

def save_learning_data(current_df):
    finished = current_df[pd.notna(pd.to_numeric(current_df['着順'], errors='coerce'))].copy()
    if finished.empty:
        return 0
    
    learning_rows = []
    for _, row in finished.iterrows():
        attrs = str(row['属性']).split(' / ')
        for attr in attrs:
            if any(k in attr for k in ['ペア', '青塗', '対称', 'リーディング']):
                key = attr
                if 'ペア' in attr:
                    match = re.search(r'ペア\((.*)\)', attr)
                    if match: key = f"ペア_{match.group(1).split(':')[-1]}"
                elif '青塗' in attr:
                    if '騎手' in attr: key = '騎手青塗'
                    elif '厩舎' in attr: key = '厩舎青塗'
                    elif '隣' in attr: key = '青塗隣'
                elif '対称' in attr:
                    if '隣' in attr: key = '対称隣'
                    else: key = '対称'
                elif 'リーディング' in attr:
                    key = 'リーディング騎手'
                
                learning_rows.append({
                    '場名': row['場名'],
                    'パターン': key,
                    '着順': row['着順'],
                    '日付': pd.Timestamp.now().strftime('%Y-%m-%d')
                })
    
    if not learning_rows:
        return 0

    new_data = pd.DataFrame(learning_rows)
    if os.path.exists(LEARNING_FILE):
        old_data = pd.read_csv(LEARNING_FILE)
        combined = pd.concat([old_data, new_data]).drop_duplicates()
    else:
        combined = new_data
        
    combined.to_csv(LEARNING_FILE, index=False)
    return len(new_data)

def get_learning_bonus(df, points_config):
    learning_df = load_learning_data()
    if learning_df.empty:
        return df
    
    learning_df['is_fukusho'] = pd.to_numeric(learning_df['着順'], errors='coerce') <= 3
    stats = learning_df.groupby(['場名', 'パターン'])['is_fukusho'].agg(['mean', 'count'])
    strong_patterns = stats[(stats['count'] >= 5) & (stats['mean'] >= 0.4)]
    
    if strong_patterns.empty:
        return df

    for idx, row in df.iterrows():
        place = row['場名']
        attrs = str(row['属性'])
        bonus = 0
        matched_Strong_patterns = []
        
        for (p_place, p_pattern), stat in strong_patterns.iterrows():
            if p_place == place:
                check_key = ""
                if "ペア_" in p_pattern: check_key = p_pattern.split("_")[1]
                elif "青塗" in p_pattern: check_key = "青塗"
                elif "対称" in p_pattern: check_key = "対称"
                elif "リーディング" in p_pattern: check_key = "リーディング"
                
                if check_key and check_key in attrs:
                    bonus += points_config['learning_bonus']
                    matched_Strong_patterns.append(f"{p_pattern}({int(stat['mean']*100)}%)")
        
        if bonus > 0:
            df.at[idx, '合計ポイント'] += bonus
            df.at[idx, '属性'] = f"🎓学習({','.join(matched_Strong_patterns)}) / " + df.at[idx, '属性']
            
    return df

# --- 4. 分析エンジン ---
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

def analyze_haichi_advanced(df_curr, df_prev=None, points_config=DEFAULT_POINTS):
    df = df_curr.copy()
    
    df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
    df.loc[(df['着順'] % 1 != 0) | (df['着順'] <= 0) | (df['着順'] > 18), '着順'] = np.nan
    
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

    # ★リーディング騎手チェック
    for idx, row in df.iterrows():
        jockey_name = str(row['騎手'])
        place_name = str(row['場名'])
        is_bonus = False
        leading_type = ""

        if any(j in jockey_name for j in JOCKEYS_SS):
            is_bonus = True
            leading_type = "SS"
        elif place_name in PLACE_KANTO and any(j in jockey_name for j in JOCKEYS_KANTO):
            is_bonus = True
            leading_type = "関東"
        elif place_name in PLACE_KANSAI and any(j in jockey_name for j in JOCKEYS_KANSAI):
            is_bonus = True
            leading_type = "関西"

        if is_bonus:
            df.at[idx, '合計ポイント'] += points_config['leading_jockey_bonus']
            df.at[idx, '属性_list'].append(f"👑リーディング({leading_type})")

    # --- A. 青塗 ---
    blue_paint_targets = []
    for category in ['騎手', '厩舎', '馬主']:
        for (place, name), group in df.groupby(['場名', category]):
            if len(group) < 2 or name == '': continue
            
            group['正循環'] = group['頭数'] + group['正番']
            group['逆循環'] = group['頭数'] + group['逆番']
            sets_list = [{r['正番'], r['逆番'], r['正循環'], r['逆循環']} for _, r in group.iterrows()]
            common_nums = set.intersection(*sets_list)
            
            if common_nums:
                num_str = list(common_nums)[0]
                pt = points_config['blue_jockey'] if category == '騎手' else points_config['blue_stable_owner']
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
                # ★修正: 親の番号をタグに含める
                tag = f"△{b['cat']}青塗隣(No.{b['正番']})"
                if tag not in current_attrs:
                    df.at[idx, '合計ポイント'] += points_config['blue_neighbor']
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
                    df.at[idx, '合計ポイント'] += points_config['sandwich_bonus']
                    df.at[idx, '属性_list'].append("🔥青塗サンドイッチ(好配置)")

    # --- C. ペア ---
    for category in ['騎手', '厩舎', '馬主']:
        for (place, name), group in df.groupby(['場名', category]):
            if len(group) < 2 or name == '': continue
            rows = group.sort_values('R').to_dict('records')
            
            pt_pair = points_config['pair_jockey'] if category == '騎手' else points_config['pair_stable_owner']
            
            for i in range(len(rows) - 1):
                r1, r2 = rows[i], rows[i+1]
                patterns = identify_pair_patterns(r1, r2)
                
                if patterns:
                    is_continuous = (abs(r2['R'] - r1['R']) == 1)
                    bonus = points_config['continuous'] if is_continuous else 0
                    
                    if 'A' in patterns:
                        bonus += points_config['pair_jockey_same_bonus'] if category == '騎手' else 0
                    
                    pattern_str = ",".join(patterns)
                    
                    for r_data in [r1, r2]:
                        idx = idx_map.get((r_data['場名'], r_data['R'], r_data['正番']))
                        if idx is not None:
                            target_r = r2['R'] if r_data['R'] == r1['R'] else r1['R']
                            tag = f"○{category}ペア({target_r}R:{pattern_str})" + ("(連)" if is_continuous else "")
                            current_list = df.at[idx, '属性_list']
                            if not any(f"○{category}ペア" in x for x in current_list):
                                df.at[idx, '合計ポイント'] += pt_pair + bonus
                                df.at[idx, '属性_list'].append(tag)
                            existing_targets = [t['R'] for t in df.at[idx, 'ペア対象_list']]
                            if target_r not in existing_targets:
                                df.at[idx, 'ペア対象_list'].append({'R': target_r, 'cat': category})

    # --- D. 対称 ---
    for (place, r), race_group in df.groupby(['場名', 'R']):
        symmetry_targets = set()
        for stable_name, stable_group in race_group.groupby('厩舎'):
            if len(stable_group) < 2 or stable_name == '': continue
            s_rows = stable_group.to_dict('records')
            for i in range(len(s_rows)):
                for j in range(i + 1, len(s_rows)):
                    s1 = {s_rows[i]['正番'], s_rows[i]['逆番'], s_rows[i]['正循環'], s_rows[i]['逆循環']}
                    s2 = {s_rows[j]['正番'], s_rows[j]['逆番'], s_rows[j]['正循環'], s_rows[j]['逆循環']}
                    if s1.intersection(s2):
                        symmetry_targets.add(s_rows[i]['正番'])
                        symmetry_targets.add(s_rows[j]['正番'])
                        
        for sym_num in symmetry_targets:
            idx = idx_map.get((place, r, sym_num))
            if idx is not None:
                if "◇厩舎対称" not in df.at[idx, '属性_list']:
                    df.at[idx, '合計ポイント'] += points_config['stable_symmetry']
                    df.at[idx, '属性_list'].append("◇厩舎対称")
            for neighbor_num in [sym_num - 1, sym_num + 1]:
                idx = idx_map.get((place, r, neighbor_num))
                if idx is not None:
                    tag = "◆厩舎対称隣"
                    if tag not in df.at[idx, '属性_list']:
                        df.at[idx, '合計ポイント'] += points_config['stable_symmetry_neighbor']
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
                        df.at[idx, '合計ポイント'] += points_config['prev_day_same_fail']
                        df.at[idx, '属性_list'].append("★前日同配置(凡走)")
                    elif prev_rank <= 3:
                        df.at[idx, '合計ポイント'] += points_config['prev_day_same_win']
                        df.at[idx, '属性_list'].append("▼前日同配置(好走)")

    # --- F. 人気加点 ---
    if '単ｵｯｽﾞ' in df.columns:
        df['人気ランク'] = df.groupby(['場名', 'R'])['単ｵｯｽﾞ'].rank(method='min')
        df.loc[df['人気ランク'] <= 5, '合計ポイント'] += points_config['odds_rank_bonus']

    df = get_learning_bonus(df, points_config)
    df['基礎ポイント'] = df['合計ポイント']
    df['属性'] = df['属性_list'].apply(lambda x: ' / '.join(x))
    return df

# --- 5. 動的ロジック & トレンド ---
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
                match = re.search(r'ペア\((.*)\)', attr)
                if match: 
                    key = f"パターン{match.group(1).split(':')[-1]}" 
                elif "青塗" in attr:
                    if "騎手" in attr: key = "騎手青塗"
                    elif "厩舎" in attr: key = "厩舎青塗"
                    elif "隣" in attr: key = "青塗隣"
                    else: key = "青塗(他)"
                elif "対称" in attr:
                    if "隣" in attr: key = "厩舎対称隣"
                    else: key = "厩舎対称"
                elif "リーディング" in attr:
                    key = "リーディング騎手"
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

def update_dynamic_points_chain(df, points_config=DEFAULT_POINTS):
    if '着順' not in df.columns: return df
    
    df['動的ポイント'] = 0.0
    df['トレンドポイント'] = 0.0
    if '基礎ポイント' in df.columns:
        df['合計ポイント'] = df['基礎ポイント']
    if '属性_list' in df.columns:
        df['属性'] = df['属性_list'].apply(lambda x: ' / '.join(x))

    bonus_map = {} 
    finished_map = {} # {idx: [要因リスト]}
    
    # --- 1. ペアのシーソー判定 ---
    for category in ['騎手', '厩舎', '馬主']:
        for (place, name), group in df.groupby(['場名', category]):
            if len(group) < 2 or name == '': continue
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
                    if curr_idx not in finished_map: finished_map[curr_idx] = []
                    finished_map[curr_idx].append(f"{category}済")
                elif not is_finished:
                    if i > 0: bonus_map[curr_idx] = bonus_map.get(curr_idx, 0) + 2.0
                
                if is_win: expectation_alive = False

    # --- 2. 青塗の終了判定 (★新機能) ---
    finished_blue_map = {} # {'騎手': {1, 5}}
    
    # 全着順をチェックして終了した青塗番号を特定
    for idx, row in df.iterrows():
        rank = pd.to_numeric(row['着順'], errors='coerce')
        if pd.isna(rank) or rank > 3: continue 

        attrs = str(row['属性'])
        # 本体: ★騎手青塗(No.1) / 隣: △騎手青塗隣(No.1)
        matches = re.findall(r'[★△](騎手|厩舎|馬主)青塗.*\(No\.(\d+)\)', attrs)
        for cat, num in matches:
            num = int(num)
            if cat not in finished_blue_map: finished_blue_map[cat] = set()
            finished_blue_map[cat].add(num)

    # 終了した青塗グループに属する馬を減点
    for idx, row in df.iterrows():
        attrs = str(row['属性'])
        matches = re.findall(r'[★△](騎手|厩舎|馬主)青塗.*\(No\.(\d+)\)', attrs)
        is_blue_finished = False
        blue_cats = []
        
        for cat, num in matches:
            num = int(num)
            if cat in finished_blue_map and num in finished_blue_map[cat]:
                is_blue_finished = True
                blue_cats.append(f"青塗({cat})済")
        
        if is_blue_finished:
            bonus_map[idx] = bonus_map.get(idx, 0) - 10.0 # 大幅減点で終了扱い
            if idx not in finished_map: finished_map[idx] = []
            finished_map[idx].extend(list(set(blue_cats)))

    # --- 3. 反映 ---
    df['終了要因'] = ""
    for idx, cats in finished_map.items():
        df.at[idx, '終了要因'] = ",".join(list(set(cats)))

    for idx, bonus in bonus_map.items():
        df.at[idx, '動的ポイント'] += bonus
        df.at[idx, '合計ポイント'] += bonus 

    trends = calculate_place_trends(df)
    st.session_state['current_trends'] = trends
    
    future_mask = pd.isna(pd.to_numeric(df['着順'], errors='coerce'))
    for idx in df[future_mask].index:
        place = df.at[idx, '場名']
        attrs = str(df.at[idx, '属性'])
        matched_trends = []
        if place in trends:
            for trend_key, data in trends[place].items():
                is_match = False
                if trend_key.startswith("パターン"):
                    p_char = trend_key.replace("パターン", "")
                    if f":{p_char})" in attrs or f"({p_char})" in attrs: is_match = True
                elif trend_key == "騎手青塗" and "騎手青塗" in attrs: is_match = True
                elif trend_key == "厩舎青塗" and "厩舎青塗" in attrs: is_match = True
                elif trend_key == "青塗隣" and "青塗隣" in attrs: is_match = True
                elif trend_key == "厩舎対称" and "厩舎対称" in attrs and "隣" not in attrs: is_match = True
                elif trend_key == "厩舎対称隣" and "厩舎対称隣" in attrs: is_match = True
                elif trend_key == "リーディング騎手" and "リーディング" in attrs: is_match = True
                if is_match: matched_trends.append(trend_key)
        
        if matched_trends:
            matched_trends = sorted(list(set(matched_trends)))
            trend_str = ",".join(matched_trends)
            bonus = len(matched_trends) * points_config['trend_bonus']
            df.at[idx, 'トレンドポイント'] = bonus
            df.at[idx, '合計ポイント'] += bonus
            df.at[idx, '属性'] = f"📈傾向({trend_str}) / " + df.at[idx, '属性']

    return df

# --- 6. UIコンポーネント ---
def render_sidebar_config():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ 分析設定")
    with st.sidebar.expander("パラメータ調整", expanded=True):
        points_config = DEFAULT_POINTS.copy()
        
        st.caption("🔻 推奨オッズ上限 (これ以上の倍率は除外)")
        odds_limit = st.slider("推奨足切りオッズ", 10.0, 100.0, 30.0, 5.0, help="このオッズ以上の馬は、どんなに配置が良くてもSS/Sランクにはなりません。")
        st.session_state['odds_limit'] = odds_limit
        
        st.divider()
        points_config['pair_jockey'] = st.slider("騎手ペア点", 0.0, 10.0, 4.0)
        points_config['leading_jockey_bonus'] = st.slider("リーディング騎手加算", 0.0, 5.0, 2.0)
        return points_config
    return DEFAULT_POINTS

def render_learning_section(full_df):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 AI学習管理")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("💾 結果を学習", help="現在の結果を保存します"):
            count = save_learning_data(full_df)
            if count > 0: st.sidebar.success(f"{count}件学習完了")
            else: st.sidebar.warning("データなし")
    
    if os.path.exists(LEARNING_FILE):
        with open(LEARNING_FILE, "rb") as f:
            st.sidebar.download_button("📥 学習データDL", f, file_name=LEARNING_FILE)
        
        uploaded_learn = st.sidebar.file_uploader("📤 学習データ復元", type=['csv'])
        if uploaded_learn:
            pd.read_csv(uploaded_learn).to_csv(LEARNING_FILE, index=False)
            st.sidebar.success("復元しました")

def render_trend_report_tab(full_df):
    st.info("📊 蓄積された学習データを元に、現在有効なパターンの統計を表示します。")
    learning_df = load_learning_data()
    if learning_df.empty:
        st.warning("⚠️ 学習データがまだありません。レース結果を入力して「結果を学習」ボタンを押してください。")
        return

    learning_df['is_win'] = pd.to_numeric(learning_df['着順'], errors='coerce') == 1
    learning_df['is_fukusho'] = pd.to_numeric(learning_df['着順'], errors='coerce') <= 3
    
    places = learning_df['場名'].unique()
    tabs = st.tabs(list(places))
    
    for tab, place in zip(tabs, places):
        with tab:
            place_data = learning_df[learning_df['場名'] == place]
            if place_data.empty: continue
            stats = place_data.groupby('パターン').agg(
                件数=('着順', 'count'),
                勝率=('is_win', 'mean'),
                複勝率=('is_fukusho', 'mean')
            ).reset_index()
            valid_stats = stats[stats['件数'] >= 3].sort_values('複勝率', ascending=False)
            
            if not valid_stats.empty:
                st.markdown(f"#### 🏆 {place}の好走パターンランキング")
                st.dataframe(
                    valid_stats.style.format({'勝率': '{:.1%}', '複勝率': '{:.1%}'}),
                    use_container_width=True
                )
            else:
                st.caption("データ収集中... (各パターン3件以上で表示されます)")

def render_race_forecast(full_df):
    st.markdown("### 🎯 厳選勝負レース (推奨買い目)")
    df = full_df.copy()
    future_mask = pd.to_numeric(df['着順'], errors='coerce').isna()
    if not future_mask.any():
        st.info("全レース終了")
        return

    odds_limit = st.session_state.get('odds_limit', 30.0)

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
        if pd.notna(odds) and odds >= odds_limit:
            return "－", f"【穴除外】オッズ{odds}倍 / " + row['属性']
        
        base_points = row.get('基礎ポイント', 0)
        is_reverse, is_jockey_origin, reverse_msg = check_blue_reverse(row, context_df)
        new_reason = row['属性']
        
        if is_reverse:
            new_reason = f"【鉄板】{reverse_msg} / " + new_reason
            if is_jockey_origin: return "◎", new_reason
            else: return "〇", new_reason 
            
        if row.get('動的ポイント', 0) > 0:
            if base_points >= 6.0: 
                return "〇", f"【激熱】直前ペア凡走+複合好配置(点数{base_points:.1f}) / {new_reason}"
            else:
                return "△", f"【注】直前ペア凡走(単独) / {new_reason}" 

        if row.get('合計ポイント', 0) >= 12.0: return "▲", new_reason
        if row.get('合計ポイント', 0) >= 8.0: return "△", new_reason
        return "－", new_reason

    places = sorted(df['場名'].unique())
    has_any = False
    
    selected_place = st.radio("推奨レースを確認する会場", places, horizontal=True, key="forecast_place")
    
    place_df = df[df['場名'] == selected_place]
    races = sorted(place_df['R'].unique())
    for r_num in races:
        race_df = place_df[place_df['R'] == r_num].copy()
        if race_df[pd.to_numeric(race_df['着順'], errors='coerce').isna()].empty:
            continue
        results = race_df.apply(lambda x: calculate_rank_and_reason(x, df), axis=1)
        race_df['ランク'] = [r[0] for r in results]
        race_df['拡張根拠'] = [r[1] for r in results]
        
        target_ranks = ['◎', '〇', '▲']
        axis_candidates = race_df[race_df['ランク'].isin(target_ranks)]
        
        if not axis_candidates.empty:
            has_any = True
            rank_map = {'◎': 3, '〇': 2, '▲': 1, '△': 0, '－': -1}
            axis_candidates['rank_score'] = axis_candidates['ランク'].map(rank_map)
            axis_horse = axis_candidates.sort_values(['rank_score', '合計ポイント'], ascending=[False, False]).iloc[0]
            
            opponents = race_df[
                (race_df['正番'] != axis_horse['正番']) &
                (race_df['動的ポイント'] >= 0) &
                ((race_df['合計ポイント'] >= 3.0) | (race_df['人気ランク'] <= 5))
            ].sort_values(['合計ポイント', '人気ランク'], ascending=[False, True]).head(4)
            
            opp_str = ",".join(opponents['正番'].astype(str).tolist())
            
            label = f"{r_num}R 【{axis_horse['ランク']}】 {axis_horse['馬名']} (軸)"
            with st.expander(label, expanded=True):
                rank_color = "red" if axis_horse['ランク'] == "◎" else "orange"
                st.markdown(f"**軸**: :{rank_color}[{axis_horse['正番']} {axis_horse['馬名']}] ({axis_horse['騎手']})")
                st.caption(f"根拠: {axis_horse['拡張根拠']}")
                st.write(f"相手: **{opp_str}**")
                        
    if not has_any:
        st.info(f"この会場に推奨馬はありません（オッズ{odds_limit}倍以下で条件合致せず）")

def render_main_tabs(full_df, points_config):
    main_tabs = st.tabs(["📋 分析メイン", "📊 傾向レポート"])
    
    with main_tabs[0]:
        places = sorted(full_df['場名'].unique())
        if not places: return
        
        selected_place = st.radio("開催会場", places, horizontal=True, key="main_place_select")
        place_df = full_df[full_df['場名'] == selected_place]
        
        st.write("---")
        
        races = sorted(place_df['R'].unique())
        selected_race = st.radio("レースを選択", races, horizontal=True, format_func=lambda x: f"{x}R", key="main_race_select")
        
        r_num = selected_race
        race_df = place_df[place_df['R'] == r_num].sort_values('正番').copy()
        
        def get_status(row):
            if row['動的ポイント'] > 0: return "🔥激熱"
            
            # 終了要因を詳細表示
            finished_cats = str(row.get('終了要因', ''))
            if finished_cats:
                if '青塗' in finished_cats: return "🛑青塗済"
                if '騎手' in finished_cats: return "🛑騎手済"
                if '厩舎' in finished_cats: return "🛑厩舎済"
                return "🛑終了"
                
            if row['動的ポイント'] < 0: return "🛑終了"
            if row['合計ポイント'] >= 10: return "⭐本命"
            return "―"
        
        def get_link(row):
            links = []
            for t in row.get('ペア対象_list', []):
                icon = "🔙" if t['R'] < row['R'] else "🔜"
                links.append(f"{icon}{t['R']}R")
            return " ".join(links)

        race_df['状態'] = race_df.apply(get_status, axis=1)
        race_df['連動'] = race_df.apply(get_link, axis=1)
        
        disabled_cols = ['枠番', '正番', '馬名', '騎手', '単ｵｯｽﾞ', '合計ポイント', 
                            '動的ポイント', 'トレンドポイント', '状態', '連動', '属性']
        
        display_cols = ['枠番', '正番', '馬名', '騎手', '単ｵｯｽﾞ', '着順', 
                        '合計ポイント', '動的ポイント', 'トレンドポイント', '状態', '連動', '属性']
        
        with st.form(key=f"form_{selected_place}_{r_num}"):
            st.caption("👇 着順入力後、更新ボタンを押してください。")
            edited = st.data_editor(
                race_df[display_cols],
                disabled=disabled_cols, 
                column_config={
                    "枠番": st.column_config.NumberColumn(width="small"),
                    "正番": st.column_config.NumberColumn(width="small"),
                    "馬名": st.column_config.TextColumn(width="medium"),
                    "騎手": st.column_config.TextColumn(width="small"),
                    "単ｵｯｽﾞ": st.column_config.NumberColumn("オッズ", format="%.1f"),
                    "着順": st.column_config.NumberColumn("着順", min_value=1, max_value=18, format="%d", help="確定した着順を入力"),
                    "合計ポイント": st.column_config.ProgressColumn("スコア", format="%.1f", min_value=-5, max_value=20),
                    "動的ポイント": st.column_config.NumberColumn("補正", format="%+.1f"),
                    "トレンドポイント": st.column_config.NumberColumn("傾向", format="%+.1f"),
                    "状態": st.column_config.TextColumn("判定", width="small"),
                    "連動": st.column_config.TextColumn("連動", width="small"),
                    "属性": st.column_config.TextColumn("根拠", width="large"),
                },
                hide_index=True,
                use_container_width=True,
                key=f"ed_{selected_place}_{r_num}"
            )
            
            submit = st.form_submit_button("データ更新 & 再計算")
            if submit:
                updates = edited.set_index('正番')['着順'].to_dict()
                full_current = st.session_state['analyzed_df']
                for idx in full_current[(full_current['場名']==selected_place) & (full_current['R']==r_num)].index:
                    n = full_current.at[idx, '正番']
                    full_current.at[idx, '着順'] = updates.get(n)
                new_df = update_dynamic_points_chain(full_current, points_config)
                st.session_state['analyzed_df'] = new_df
                st.rerun()

    with main_tabs[1]:
        render_trend_report_tab(full_df)

# --- 7. メイン処理フロー ---
def main():
    st.sidebar.title("🏇 設定・データ")
    
    points_config = render_sidebar_config()
    
    st.sidebar.subheader("💾 途中経過の読み込み")
    uploaded_progress = st.sidebar.file_uploader("保存したCSVを読み込む", type=['csv'], key="progress")
    st.sidebar.subheader("🆕 新規分析")
    uploaded_curr = st.sidebar.file_uploader("当日出馬表 (必須)", type=['xlsx', 'csv'], key="curr")
    uploaded_prev = st.sidebar.file_uploader("前日出馬表 (土日連動用)", type=['xlsx', 'csv'], key="prev")
    
    full_df = None

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
                    df = analyze_haichi_advanced(df_curr, df_prev, points_config)
                    df = update_dynamic_points_chain(df, points_config)
                st.session_state['analyzed_df'] = df
                st.session_state['last_session_key'] = session_key
            else:
                st.error(f"読み込みエラー: {msg1}")
    
    if 'analyzed_df' in st.session_state:
        full_df = st.session_state['analyzed_df']
        
        # 強制リセット
        full_df['着順'] = pd.to_numeric(full_df['着順'], errors='coerce')
        full_df.loc[(full_df['着順'] <= 0) | (full_df['着順'] > 18), '着順'] = np.nan
        st.session_state['analyzed_df'] = full_df 
        
        render_learning_section(full_df)
        
        st.sidebar.markdown("---")
        if st.sidebar.button("⚠️ 着順データを全リセット", help="『終了』と誤判定される場合に押してください。"):
            full_df['着順'] = np.nan
            full_df = update_dynamic_points_chain(full_df, points_config)
            st.session_state['analyzed_df'] = full_df
            st.success("着順データを全てリセットしました！")
            st.rerun()

    if full_df is not None:
        csv = full_df.to_csv(index=False).encode('utf-8-sig')
        st.sidebar.download_button(
            "💾 分析データを保存 (CSV)",
            csv,
            "haichi_analysis_save.csv",
            "text/csv",
            help="現在の状態を保存"
        )
        
        render_main_tabs(full_df, points_config)
        st.divider()
        # render_trend_main()
        render_race_forecast(full_df)
    else:
        st.info("👈 サイドバーからデータをアップロードしてください。\n(例: 260104全出走馬.csv)")

if __name__ == "__main__":
    main()
