def render_recommendations(full_df):
    """画面上部の推奨枠（ランク付き・タブ表示版）"""
    st.markdown("### 🏆 特選推奨馬 (AI厳選ランク)")
    df = full_df.copy()
    
    # 1. 抽出条件
    future = df[pd.to_numeric(df['着順'], errors='coerce').isna()]
    cond_odds = pd.to_numeric(future['単ｵｯｽﾞ'], errors='coerce').fillna(0) <= 49.9
    
    target_df = future[
        ((future['動的ポイント'] > 0) | (future['合計ポイント'] >= 7.0)) &
        cond_odds & (future['動的ポイント'] >= 0)
    ].copy()
    
    if target_df.empty:
        st.info("現在、特選推奨条件に合致する馬はいません。")
        return

    # 2. ランク判定ロジック
    def get_rank(row):
        # シーソー理論で「狙いたいケース」になっている馬を最上位 [cite: 130]
        if row['動的ポイント'] > 0:
            return "S"
        # 基礎点が圧倒的に高い馬
        if row['合計ポイント'] >= 10.0:
            return "A"
        return "B"

    target_df['ランク'] = target_df.apply(get_rank, axis=1)

    # 3. 買い目生成
    tickets = []
    for _, row in target_df.iterrows():
        race_df = full_df[(full_df['場名'] == row['場名']) & (full_df['R'] == row['R'])]
        tickets.append(generate_betting_ticket(row, race_df))
    target_df['推奨買い目'] = tickets

    # 4. タブ表示 (ランク順にソートして表示)
    places = sorted(target_df['場名'].unique())
    p_tabs = st.tabs(places)
    
    for p_tab, place in zip(p_tabs, places):
        with p_tab:
            place_rec_df = target_df[target_df['場名'] == place]
            races = sorted(place_rec_df['R'].unique())
            r_tabs = st.tabs([f"{r}R" for r in races])
            
            for r_tab, r_num in zip(r_tabs, races):
                with r_tab:
                    # ソート順: ランク(S>A>B) > 合計ポイント(降順)
                    # ランクを数値化してソートするテクニック
                    rank_map = {"S": 3, "A": 2, "B": 1}
                    place_rec_df['rank_num'] = place_rec_df['ランク'].map(rank_map)
                    
                    disp_df = place_rec_df[place_rec_df['R'] == r_num].sort_values(
                        ['rank_num', '合計ポイント'], ascending=[False, False]
                    )
                    
                    st.dataframe(
                        disp_df[['ランク', '正番', '馬名', '単ｵｯｽﾞ', '合計ポイント', '動的ポイント', '推奨買い目', '属性']],
                        column_config={
                            "ランク": st.column_config.TextColumn("ランク", width="small", help="S:激熱(狙い目), A:本命, B:推奨"),
                            "正番": st.column_config.NumberColumn("番", width="small", format="%d"),
                            "合計ポイント": st.column_config.ProgressColumn("総合評価", min_value=0, max_value=20, format="%.1f"),
                            "動的ポイント": st.column_config.NumberColumn("補正", format="%+.1f", help="プラス値は直前のペア凡走によるチャンス加点"),
                            "推奨買い目": st.column_config.TextColumn("🎫 買い目", width="large"),
                            "属性": st.column_config.TextColumn("根拠", width="medium")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
    st.markdown("---")
