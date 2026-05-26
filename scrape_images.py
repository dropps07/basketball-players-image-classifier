from bing_image_downloader import downloader

players = {
    "lebron_james":    "LeBron James face",
    "michael_jordan":  "Michael Jordan face",
    "kobe_bryant":     "Kobe Bryant face",
    "stephen_curry":   "Stephen Curry face",
    "shaquille_oneal": "Shaquille ONeal face",
}

for folder, query in players.items():
    print(f"\nScraping: {query}...")
    downloader.download(
        query,
        limit=150,
        output_dir="basketball_dataset",
        adult_filter_off=True,
        force_replace=False,
        timeout=60,
        verbose=True
    )