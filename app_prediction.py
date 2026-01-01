import streamlit as st
import pandas as pd
import numpy as np
import re

# --- 1. 基本設定 & ユーティリティ ---
st.set_page_config(page_title="配置馬券術AI分析システム", layout="wide")

# ポイント配分設定 (資料に基づく)
HAICHI_POINTS = {
    'blue_paint': 2.0,       # 青塗 (騎手・厩舎・馬主)
    'blue_neighbor': 2.0,    # 青塗隣
    'stable_symmetry': 2.0,  # 厩舎対称配置
    'pair_exist': 1.0,       # 配置チェック(ペア)がある
    'continuous': 1.0,       # 連続レース配置
    'odds_rank_bonus': 1.0,  # 1~5番人気
    'prev_day_same_fail': 1.0, # 前日同R同配置で凡走
    'prev_day_same_win': -1.0, # 前日同R同配置で好走
}

def to_half_width(text):
    """全角数字を半角に変換"""
    if pd.isna(text): return text
    text = str(text)
    table = str.maketrans('０１２３４５６７８９．', '0123456789.')
    return re.sub(r'[^\d\.]', '', text.translate(table))

def normalize_name(x):
    """名前の正規化（空白削除など）"""
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
        
        # カラム名探索
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
            '着': '着順', '着順': '着順' # 既存の着順があれば維持
        }
        df = df.rename(columns=name_map)
        
        # 必須カラム確保
        ensure_cols = ['場名', 'R', '馬名', '正番', '騎手', '厩舎', '馬主', '単ｵｯｽﾞ', '着順']
        for col in ensure_cols:
            if col not in df.columns: df[col] = np.nan

        # 数値変換
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

# --- 3. 分析エンジン (静的スコア計算) ---
def analyze_haichi_advanced(df_curr, df_prev=None):
    df = df_curr.copy()
    
    # 基本4数字の計算
    max_umaban = df.groupby(['場名', 'R'])['正番'].transform('max')
    df['頭数'] = max_umaban.fillna(16).astype(int)
    df['逆番'] = (df['頭数'] + 1) - df['正番']
    df['正循環'] = df['頭数'] + df['正番']
    df['逆循環'] = df['頭数'] + df['逆番']
    
    for c in ['正番', '逆番', '正循環', '逆循環']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

    # 結果格納用 (既存データがあれば保持)
    if '合計ポイント' not in df.columns: df['合計ポイント'] = 0.0
    if '動的ポイント' not in df.columns: df['動的ポイント'] = 0.0
    
    # 属性リストの初期化 (文字列からリストへの復元など)
    if '属性_list' not in df.columns:
        df['属性_list'] = [[] for _ in range(len(df))]
    else:
        # CSV読み込み時は文字列になっている可能性があるためリセット推奨
        # 今回は再計算するためリセットします
        df['合計ポイント'] = 0.0 # ポイントは再計算
        df['属性_list'] = [[] for _ in range(len(df))]
        
    df['ペア対象_list'] = [[] for _ in range(len(df))] 

    idx_map = {(row['場名'], row['R'], row['正番']): i for i, row in df.iterrows()}

    # --- A. 青塗 (Blue Paint) ---
    blue_paint_targets = []
    for category in ['騎手', '厩舎', '馬主']:
        for (place, name), group in df.groupby(['場名', category]):
            if len(group) < 2: continue
            sets_list = [{r['正番'], r['逆番'], r['正循環'], r['逆循環']} for _, r in group.iterrows()]
            common_nums = set.intersection(*sets_list)
            
            if common_nums:
                num_str = list(common_nums)[0]
                for _, row in group.iterrows():
                    idx = idx_map.get((row['場名'], row['R'], row['正番']))
                    if idx is not None:
                        df.at[idx, '合計ポイント'] += HAICHI_POINTS['blue_paint']
                        df.at[idx, '属性_list'].append(f"★{category}青塗(No.{num_str})")
                        blue_paint_targets.append({'場名': place, 'R': row['R'], '正番': row['正番'], 'cat': category})

    # --- B. 青塗隣 (Neighbor) ---
    for b in blue_paint_targets:
        for neighbor_num in [b['正番'] - 1, b['正番'] + 1]:
            idx = idx_map.get((b['場名'], b['R'], neighbor_num))
            if idx is not None:
                current_attrs = df.at[idx, '属性_list']
                tag = f"△{b['cat']}青塗隣"
                if tag not in current_attrs:
                    df.at[idx, '合計ポイント'] += HAICHI_POINTS['blue_neighbor']
                    df.at[idx, '属性_list'].append(tag)

    # --- C. ペア & 連続レース ---
    for category in ['騎手', '厩舎', '馬主']:
        for (place, name), group in df.groupby(['場名', category]):
            if len(group) < 2: continue
            rows = group.sort_values('R').to_dict('records')
            
            for i in range(len(rows) - 1):
                r1, r2 = rows[i], rows[i+1]
                nums1 = {r1['正番'], r1['逆番'], r1['正循環'], r1['逆循環']}
                nums2 = {r2['正番'], r2['逆番'], r2['正循環'], r2['逆循環']}
                
                if nums1.intersection(nums2):
                    is_continuous = (r2['R'] - r1['R'] == 1)
                    bonus = HAICHI_POINTS['continuous'] if is_continuous else 0
                    
                    for r_data in [r1, r2]:
                        idx = idx_map.get((r_data['場名'], r_data['R'], r_data['正番']))
                        if idx is not None:
                            tag = f"○{category}ペア" + ("(連続)" if is_continuous else "")
                            if tag not in df.at[idx, '属性_list']:
                                df.at[idx, '合計ポイント'] += HAICHI_POINTS['pair_exist'] + bonus
                                df.at[idx, '属性_list'].append(tag)
                            target_r = r2['R'] if r_data['R'] == r1['R'] else r1['R']
                            df.at[idx, 'ペア対象_list'].append({'R': target_r, 'cat': category})

    # --- D. 厩舎対称配置 ---
    for (place, r), race_group in df.groupby(['場名', 'R']):
        for stable_name, stable_group in race_group.groupby('厩舎'):
            if len(stable_group) < 2: continue
            s_rows = stable_group.to_dict('records')
            has_symmetry = False
            for i in range(len(s_rows)):
                for j in range(i + 1, len(s_rows)):
                    s1 = {s_rows[i]['正番'], s_rows[i]['逆番'], s_rows[i]['正循環'], s_rows[i]['逆循環']}
                    s2 = {s_rows[j]['正番'], s_rows[j]['逆番'], s_rows[j]['正循環'], s_rows[j]['逆循環']}
                    if s1.intersection(s2):
                        has_symmetry = True
            if has_symmetry:
                for idx_s, _ in stable_group.iterrows():
                    if "◇厩舎対称" not in df.at[idx_s, '属性_list']:
                        df.at[idx_s, '合計ポイント'] += HAICHI_POINTS['stable_symmetry']
                        df.at[idx_s, '属性_list'].append("◇厩舎対称")

    # [cite_start]--- E. 前日データの横比較 (New!) [cite: 87] ---
    if df_prev is not None and not df_prev.empty:
        # 前日データの(場名, R, 正番) -> 着順マップを作成
        # ※場名は同じ前提で比較しますが、異なる場合は調整が必要
        prev_map = {}
        for _, row in df_prev.iterrows():
            # 正番だけでなく、逆番なども一致すれば...となりますが、資料では「同じ位置(枠・番)」を指すことが多い
            # ここではシンプルに「同R・同番」を比較対象とします
            key = (row['R'], row['正番']) 
            prev_map[key] = pd.to_numeric(row['着順'], errors='coerce')

        for idx, row in df.iterrows():
            key = (row['R'], row['正番'])
            if key in prev_map:
                prev_rank = prev_map[key]
                if pd.notna(prev_rank):
                    if prev_rank > 3: # 凡走
                        df.at[idx, '合計ポイント'] += HAICHI_POINTS['prev_day_same_fail']
                        df.at[idx, '属性_list'].append("★前日同配置(凡走)")
                    elif prev_rank <= 3: # 好走
                        df.at[idx, '合計ポイント'] += HAICHI_POINTS['prev_day_same_win'] # 減点
                        df.at[idx, '属性_list'].append("▼前日同配置(好走)")

    # --- F. オッズ加点 ---
    if '単ｵｯｽﾞ' in df.columns:
        df['人気ランク'] = df.groupby(['場名', 'R'])['単ｵｯｽﾞ'].rank(method='min')
        df.loc[df['人気ランク'] <= 5, '合計ポイント'] += HAICHI_POINTS['odds_rank_bonus']

    df['属性'] = df['属性_list'].apply(lambda x: ' / '.join(x))
    return df

# --- 4. 動的ロジック (シーソー/玉突き連動) ---
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
                    if i > 0: 
                        bonus_map[curr_idx] = bonus_map.get(curr_idx, 0) + 2.0
                
                if is_win: expectation_alive = False
                
    for idx, bonus in bonus_map.items():
        df.at[idx, '動的ポイント'] += bonus
        df.at[idx, '合計ポイント'] += bonus 

    return df

# --- 5. 買い目生成 ---
def generate_betting_ticket(row, race_df):
    recs = []
    odds = pd.to_numeric(row.get('単ｵｯｽﾞ'), errors='coerce')
    if pd.isna(odds): odds = 0.0
    recs.append("単複" if odds >= 10.0 else "単勝")
    
    opponents = race_df[race_df['正番'] != row['正番']].copy()
    opponents['単ｵｯｽﾞ'] = pd.to_numeric(opponents['単ｵｯｽﾞ'], errors='coerce').fillna(999.9)
    opponents['人気順'] = opponents['単ｵｯｽﾞ'].rank(method='min')
    
    targets = opponents[
        ((opponents['合計ポイント'] >= 3.0) & (opponents['動的ポイント'] >= 0)) |
        (opponents['人気順'] <= 5)
    ].drop_duplicates(subset='正番')
    
    targets = targets.sort_values(['合計ポイント', '人気順'], ascending=[False, True]).head(5)
    t_str = ",".join(targets['正番'].astype(int).astype(str).tolist())
    
    if t_str:
        recs.append(f"ワイド {row['正番']}-{t_str}")
        if row.get('合計ポイント', 0) >= 7.0 or row.get('動的ポイント', 0) > 0:
            count = len(targets)
            points = count * (count - 1) // 2
            recs.append(f"3連複 {row['正番']}-{t_str} ({points}点)")
            
    return " / ".join(recs)

# --- 6. UIコンポーネント ---
def render_recommendations(full_df):
    st.markdown("### 🏆 特選推奨馬 (AI厳選)")
    df = full_df.copy()
    future = df[pd.to_numeric(df['着順'], errors='coerce').isna()]
    cond_odds = pd.to_numeric(future['単ｵｯｽﾞ'], errors='coerce').fillna(0) <= 49.9
    
    targets = future[
        ((future['動的ポイント'] > 0) | (future['合計ポイント'] >= 7.0)) &
        cond_odds & (future['動的ポイント'] >= 0)
    ].sort_values('合計ポイント', ascending=False)
    
    if targets.empty:
        st.info("現在、特選推奨条件に合致する馬はいません。")
        return

    tickets = []
    for _, row in targets.iterrows():
        race_df = full_df[(full_df['場名']==row['場名']) & (full_df['R']==row['R'])]
        tickets.append(generate_betting_ticket(row, race_df))
    targets['推奨買い目'] = tickets
    targets['レース'] = targets['場名'] + targets['R'].astype(str) + "R"

    st.dataframe(
        targets[['レース', '正番', '馬名', '単ｵｯｽﾞ', '合計ポイント', '動的ポイント', '推奨買い目', '属性']],
        column_config={
            "合計ポイント": st.column_config.ProgressColumn("総合評価", min_value=0, max_value=20, format="%.1f"),
            "動的ポイント": st.column_config.NumberColumn("補正", format="%+.1f", help="プラスはチャンス"),
            "推奨買い目": st.column_config.TextColumn("🎫 買い目", width="medium"),
            "属性": st.column_config.TextColumn("根拠", width="medium")
        },
        hide_index=True, use_container_width=True
    )
    st.markdown("---")

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
                                    "馬名": st.column_config.TextColumn(disabled=True),
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
                        disp[['枠番', '正番', '馬名', '騎手', '単ｵｯｽﾞ', '合計ポイント', '動的ポイント', '状態', '連動', '属性']],
                        column_config={
                            "合計ポイント": st.column_config.ProgressColumn("スコア", format="%.1f", min_value=-5, max_value=20),
                            "動的ポイント": st.column_config.NumberColumn("補正", format="%+.1f"),
                            "状態": st.column_config.TextColumn("判定", width="small"),
                            "属性": st.column_config.TextColumn("根拠", width="large"),
                        },
                        hide_index=True, use_container_width=True
                    )

# --- 7. メイン処理フロー ---
def main():
    st.sidebar.title("🏇 設定・データ")
    
    # A. 続きから始める場合 (Save/Load)
    st.sidebar.subheader("💾 途中経過の読み込み")
    uploaded_progress = st.sidebar.file_uploader("保存したCSVを読み込む", type=['csv'], key="progress")
    
    # B. 新規分析の場合
    st.sidebar.subheader("🆕 新規分析")
    uploaded_curr = st.sidebar.file_uploader("当日出馬表 (必須)", type=['xlsx', 'csv'], key="curr")
    uploaded_prev = st.sidebar.file_uploader("前日出馬表 (土日連動用)", type=['xlsx', 'csv'], key="prev")

    # ロード処理の分岐
    if uploaded_progress:
        # 続きから
        try:
            df = pd.read_csv(uploaded_progress)
            # リスト等の復元が必要な場合はここで行うが、今回は文字列のまま表示でもOK
            # 分析用には再計算するため、最低限のカラムがあればよい
            st.session_state['analyzed_df'] = df
            st.sidebar.success("復元しました！")
        except Exception as e:
            st.sidebar.error(f"復元エラー: {e}")
            
    elif uploaded_curr:
        # 新規読み込み (ファイルが変わった時のみ実行)
        curr_name = uploaded_curr.name
        prev_name = uploaded_prev.name if uploaded_prev else "none"
        session_key = f"{curr_name}_{prev_name}"
        
        if 'last_session_key' not in st.session_state or st.session_state['last_session_key'] != session_key:
            df_curr, msg1 = load_data(uploaded_curr)
            df_prev, msg2 = (None, "")
            
            if uploaded_prev:
                df_prev, msg2 = load_data(uploaded_prev)
            
            if msg1 == "success":
                # 分析実行
                with st.spinner("AI分析を実行中..."):
                    df = analyze_haichi_advanced(df_curr, df_prev)
                    # 初期状態での動的計算
                    df = update_dynamic_points_chain(df)
                    
                st.session_state['analyzed_df'] = df
                st.session_state['last_session_key'] = session_key
            else:
                st.error(f"読み込みエラー: {msg1}")

    # メイン画面表示
    if 'analyzed_df' in st.session_state:
        full_df = st.session_state['analyzed_df']
        
        # 保存ボタン (現在の状態をCSVでダウンロード)
        csv = full_df.to_csv(index=False).encode('utf-8-sig')
        st.sidebar.download_button(
            "💾 分析データを保存 (CSV)",
            csv,
            "haichi_analysis_save.csv",
            "text/csv",
            help="現在の入力内容や分析結果を保存します。次回「途中経過の読み込み」から再開できます。"
        )
        
        # 1. 推奨馬表示
        render_recommendations(full_df)
        
        # 2. タブ詳細表示
        render_main_tabs(full_df)
    else:
        st.info("👈 サイドバーからデータをアップロードしてください。")
        st.markdown("""
        **使い方:**
        1. **当日出馬表** に今日のエクセル/CSVを入れます。
        2. (土日の場合) [cite_start]**前日出馬表** に昨日の結果入りエクセルを入れると、精度がアップします[cite: 87]。
        3. 途中で中断したいときは **「分析データを保存」** ボタンでファイルを保存してください。
        """)

if __name__ == "__main__":
    main()
