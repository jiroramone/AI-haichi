def render_race_forecast(full_df):
    """
    レース単位で「軸馬1頭」＋「相手厳選」を行い、
    点数を極限まで絞った買い目を提案する機能
    """
    st.markdown("### 🎯 厳選勝負レース (推奨買い目)")
    
    # 全データをコピー
    df = full_df.copy()
    
    # ---------------------------------------------------------
    # 1. 下準備: 全馬にランクとオッズ逆転判定を付与
    # ---------------------------------------------------------
    # 未来のレースのみ対象
    future_mask = pd.to_numeric(df['着順'], errors='coerce').isna()
    if not future_mask.any():
        st.info("全てのレースが終了しています。")
        return

    # 青塗逆転判定関数 (前回と同じ)
    def check_blue_reverse(row, context_df):
        my_attrs = str(row.get('属性', ''))
        if '△' not in my_attrs or '青塗隣' not in my_attrs: return False
        my_num = row['正番']
        my_odds = pd.to_numeric(row['単ｵｯｽﾞ'], errors='coerce')
        if pd.isna(my_odds): return False
        
        race_df = context_df[(context_df['場名'] == row['場名']) & (context_df['R'] == row['R'])]
        for offset in [-1, 1]:
            neighbor_num = my_num + offset
            n_row = race_df[race_df['正番'] == neighbor_num]
            if n_row.empty: continue
            n_attrs = str(n_row.iloc[0].get('属性', ''))
            if '★' in n_attrs and '青塗' in n_attrs:
                n_odds = pd.to_numeric(n_row.iloc[0]['単ｵｯｽﾞ'], errors='coerce')
                if pd.notna(n_odds) and my_odds < n_odds: return True
        return False

    # ランク判定
    def calculate_rank(row, context_df):
        # オッズ50倍以上は推奨対象外
        odds = pd.to_numeric(row['単ｵｯｽﾞ'], errors='coerce')
        if pd.isna(odds) or odds > 49.9: return "C"
        
        # 死に目（動的マイナス）は除外
        if row.get('動的ポイント', 0) < 0: return "C"

        # SS: 青塗隣かつオッズ逆転
        if check_blue_reverse(row, context_df): return "SS"
        # S: シーソー激熱
        if row.get('動的ポイント', 0) > 0: return "S"
        # A: 基礎点が高い
        if row.get('合計ポイント', 0) >= 10.0: return "A"
        # B: 水準以上
        if row.get('合計ポイント', 0) >= 7.0: return "B"
        
        return "C"

    # 計算負荷を下げるため、必要なレースだけループ計算
    # (実際は全行計算しても早いが、念のため)
    df['ランク'] = 'C'
    
    # ---------------------------------------------------------
    # 2. レースごとに戦略を構築
    # ---------------------------------------------------------
    # 会場・レースでグルーピング
    places = sorted(df['場名'].unique())
    
    # 表示するレースがあったかどうかのフラグ
    has_recommendation = False
    
    # タブで会場を分ける
    p_tabs = st.tabs(places)
    
    for p_tab, place in zip(p_tabs, places):
        with p_tab:
            place_df = df[df['場名'] == place]
            races = sorted(place_df['R'].unique())
            
            # 各レースをチェック
            for r_num in races:
                race_df = place_df[place_df['R'] == r_num].copy()
                
                # まだ終わっていないレースか確認
                if not race_df[pd.to_numeric(race_df['着順'], errors='coerce').isna()].empty:
                    
                    # ランク計算
                    race_df['ランク'] = race_df.apply(lambda x: calculate_rank(x, df), axis=1)
                    
                    # 軸馬候補 (S以上がいるレースのみ推奨とする)
                    # 優先度: SS > S > A (AはSS/Sがいない場合の抑えだが、今回は厳選のためSS/Sのみ)
                    axis_candidates = race_df[race_df['ランク'].isin(['SS', 'S'])]
                    
                    if not axis_candidates.empty:
                        has_recommendation = True
                        
                        # 最も良い1頭を軸に選定 (ランク > スコア順)
                        rank_map = {'SS': 3, 'S': 2}
                        axis_candidates['rank_score'] = axis_candidates['ランク'].map(rank_map)
                        axis_horse = axis_candidates.sort_values(
                            ['rank_score', '合計ポイント'], ascending=[False, False]
                        ).iloc[0]
                        
                        # --- 買い目の構築 (点数を絞る) ---
                        
                        # 相手候補 (Opponents)
                        # 条件: (スコア3点以上 OR 人気1-5番) かつ (死に目ではない)
                        opponents = race_df[
                            (race_df['正番'] != axis_horse['正番']) &
                            (race_df['動的ポイント'] >= 0) & # 死に目は買わない
                            (
                                (race_df['合計ポイント'] >= 3.0) |
                                (race_df['人気ランク'] <= 5) # 人気ブログの上位
                            )
                        ].copy()
                        
                        # 相手を強力な順にソート (スコア > 人気)
                        opponents = opponents.sort_values(['合計ポイント', '人気ランク'], ascending=[False, True])
                        
                        # 相手を絞る (最大4頭)
                        # 軸がSSランク(鉄板)なら相手を少し広げてもいいが、基本は絞る
                        final_opponents = opponents.head(4)
                        opponent_nums = final_opponents['正番'].astype(str).tolist()
                        opponent_str = ",".join(opponent_nums)
                        
                        # --- 表示エリアの作成 ---
                        with st.container():
                            # ヘッダー (例: 東京 11R [SS] 軸:武豊)
                            rank_color = "red" if axis_horse['ランク'] == "SS" else "orange"
                            st.markdown(
                                f"##### :{rank_color}[【{axis_horse['ランク']}】 {place} {r_num}R] 軸: {axis_horse['正番']} {axis_horse['馬名']} ({axis_horse['騎手']})"
                            )
                            
                            cols = st.columns([2, 3])
                            
                            with cols[0]:
                                # 軸馬データ
                                st.info(f"**推奨根拠**: {axis_horse['属性']}")
                                st.write(f"単勝オッズ: **{axis_horse['単ｵｯｽﾞ']}**倍")
                                st.write(f"総合スコア: **{axis_horse['合計ポイント']}** (補正 {axis_horse['動的ポイント']:+.1f})")
                                
                            with cols[1]:
                                # 買い目提案
                                st.write("**🎫 推奨買い目 (資金配分推奨)**")
                                
                                bets = []
                                axis_num = axis_horse['正番']
                                axis_odds = pd.to_numeric(axis_horse['単ｵｯｽﾞ'], errors='coerce')
                                
                                # 1. 単勝 (オッズがついている場合)
                                if axis_odds >= 3.0:
                                    bets.append(f"- **単勝**: {axis_num} (本線)")
                                elif axis_odds < 3.0:
                                    bets.append(f"- 単勝: {axis_num} (見送り/安すぎ)")
                                
                                # 2. ワイド (点数を絞る)
                                if opponent_nums:
                                    # 相手が1頭だけなら1点、複数なら流し
                                    bets.append(f"- **ワイド**: {axis_num} － {opponent_str} ({len(opponent_nums)}点)")
                                    
                                    # 3. 3連複 (SSランク または オッズ妙味がある場合のみ)
                                    if axis_horse['ランク'] == 'SS' or axis_odds >= 10.0:
                                        # 1頭軸流し
                                        points = len(opponent_nums) * (len(opponent_nums)-1) // 2
                                        bets.append(f"- **3連複**: {axis_num} － {opponent_str} ({points}点ボーナス)")
                                else:
                                    bets.append("- 相手不在のため単勝のみ推奨")
                                
                                for bet in bets:
                                    st.write(bet)
                            
                            st.markdown("---")

    if not has_recommendation:
        st.info("現在、厳選条件（SS/Sランク）に合致する勝負レースはありません。")
