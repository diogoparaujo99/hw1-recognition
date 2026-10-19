"""Capture the actual TensorBoard image dashboard for the inference deliverable."""
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:6006')
    parser.add_argument('--output',type=Path,default=ROOT/'report/figures/q3_inference.png')
    args=parser.parse_args()
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1900,'height':1250},device_scale_factor=1,timezone_id='America/New_York')
        page.goto(args.url+'/#images',wait_until='networkidle')
        page.wait_for_timeout(4000)
        # Filter tags to retain inference images and omit ground-truth previews.
        page.locator('input').nth(0).fill('fcos_inference')
        field=page.locator('input').nth(1)
        field.fill('^test_images$')
        field.press('Enter')
        page.get_by_text('Show actual image size',exact=True).click()
        page.wait_for_timeout(3000)
        page.screenshot(path=str(args.output),full_page=True)
        browser.close()
    print(args.output)


if __name__=='__main__':
    main()
