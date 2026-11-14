"""Capture documentation screenshots of the running dashboard.

Usage:
    streamlit run app.py --server.port 8599 &
    python scripts/take_screenshots.py [--url http://localhost:8599] [--browser /path/to/chromium]

Uses Playwright. If you have not run `playwright install chromium`, pass --browser with the
path of any Chromium-based browser (Chrome, Edge, Brave).
"""

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "screenshots"
OUT.mkdir(exist_ok=True)


def wait_ready(page, extra=2.5):
    page.wait_for_selector('[data-testid="stApp"]')
    # wait until Streamlit stops showing its "running" indicator
    for _ in range(60):
        if page.locator('[data-testid="stStatusWidget"]').count() == 0:
            break
        time.sleep(0.5)
    time.sleep(extra)


def ask(page, text):
    box = page.get_by_placeholder("e.g. What was the median age")
    box.fill(text)
    box.press("Enter")
    wait_ready(page, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8599")
    ap.add_argument("--browser", default=None)
    args = ap.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.browser, headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 950}, device_scale_factor=1.5)

        # 1. Dashboard - first screen (KPIs + first charts)
        page.goto(args.url)
        page.wait_for_selector(".js-plotly-plot", timeout=60000)
        wait_ready(page, 4)
        page.screenshot(path=OUT / "01_dashboard_top.png")

        # 2. Whole dashboard: make the viewport tall enough to show every chart
        page.set_viewport_size({"width": 1500, "height": 2350})
        wait_ready(page, 3)
        page.screenshot(path=OUT / "02_dashboard_full.png")

        # 3. Filters in action: only 3rd class
        page.set_viewport_size({"width": 1500, "height": 950})
        sidebar = page.locator('[data-testid="stSidebar"]')
        for cls in ["1st", "2nd"]:
            sidebar.get_by_role("button", name=f"Remove {cls}").first.click()
            wait_ready(page, 1.5)
        wait_ready(page, 2)
        page.screenshot(path=OUT / "03_filters_third_class.png")
        page.goto(args.url)   # reset filters
        page.wait_for_selector(".js-plotly-plot", timeout=60000)
        wait_ready(page, 3)

        # 4. Chatbot tab
        page.get_by_role("tab", name="💬 Ask the Data (Chatbot)").click()
        wait_ready(page, 1.5)
        page.screenshot(path=OUT / "04_chatbot_welcome.png")

        page.get_by_role("button", name="Survival rate by class and sex").click()
        wait_ready(page, 2)
        page.screenshot(path=OUT / "05_chatbot_survival_by_class_sex.png")

        ask(page, "Which columns are correlated with survival?")
        page.screenshot(path=OUT / "06_chatbot_correlation.png")

        ask(page, "How many children in third class survived?")
        ask(page, "Describe the fare column")
        page.screenshot(path=OUT / "07_chatbot_counts_and_stats.png")

        # 5. Data & dictionary tab
        page.get_by_role("tab", name="🗂️ Data & Dictionary").click()
        wait_ready(page, 2)
        page.screenshot(path=OUT / "08_data_dictionary.png")

        browser.close()
    print("Saved screenshots to", OUT)


if __name__ == "__main__":
    main()
