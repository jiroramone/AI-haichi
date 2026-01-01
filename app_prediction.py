def render_recommendations(full_df):
    """画面上部の推奨枠（会場・レース別タブ表示版）"""
    st.markdown("### 🏆 特選推奨馬 (AI厳選)")
    df = full_df.copy()
    
    # --- 1. 推奨馬の抽出ロジック (変更なし) ---
    # まだ走っていない(着順なし) & 50倍以下
    future = df[pd.to_numeric(df['着順'], errors='coerce').isna()]
    cond_odds = pd.to_numeric(future['単ｵｯｽﾞ'], errors='coerce').fillna(0) <= 49.9
    
    # 激熱(動的プラス) または 高スコア(7点以上) かつ 死に目ではない
    target_df = future[
        ((future['動的ポイント'] > 0) | (future['合計ポイント'] >= 7.0)) &
        cond_odds & (future['動的ポイント'] >= 0)
    ].copy()
    
    if target_df.empty:
        st.info("現在、特選推奨条件に合致する馬はいません。")
        return

    # --- 2. 買い目生成 (まとめて計算) ---
    tickets = []
    for _, row in target_df.iterrows():
        race_df = full_df[(full_df['場名'] == row['場名']) & (full_df['R'] == row['R'])]
        tickets.append(generate_betting_ticket(row, race_df))
    target_df['推奨買い目'] = tickets

    # --- 3. タブ表示ロジック (ここを変更) ---
    
    # 推奨馬がいる会場だけをリストアップ
    places = sorted(target_df['場名'].unique())
    
    # A. 会場タブ
    p_tabs = st.tabs(places)
    
    for p_tab, place in zip(p_tabs, places):
        with p_tab:
            # その会場の推奨馬データ
            place_rec_df = target_df[target_df['場名'] == place]
            
            # 推奨馬がいるレース番号を取得
            races = sorted(place_rec_df['R'].unique())
            
            # B. レースタブ (推奨馬がいるレースのみ作成)
            r_tabs = st.tabs([f"{r}R" for r in races])
            
            for r_tab, r_num in zip(r_tabs, races):
                with r_tab:
                    # そのレースの推奨馬を表示
                    disp_df = place_rec_df[place_rec_df['R'] == r_num].sort_values('合計ポイント', ascending=False)
                    
                    st.dataframe(
                        disp_df[['正番', '馬名', '単ｵｯｽﾞ', '合計ポイント', '動的ポイント', '推奨買い目', '属性']],
                        column_config={
                            "正番": st.column_config.NumberColumn("番", width="small", format="%d"),
                            "合計ポイント": st.column_config.ProgressColumn("総合評価", min_value=0, max_value=20, format="%.1f"),
                            "動的ポイント": st.column_config.NumberColumn("補正", format="%+.1f", help="プラスはチャンス"),
                            "推奨買い目": st.column_config.TextColumn("🎫 買い目", width="large"),
                            "属性": st.column_config.TextColumn("根拠", width="medium")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
    st.markdown("---")
