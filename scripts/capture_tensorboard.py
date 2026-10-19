"""Capture TensorBoard's own histogram and inference displays."""
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def open_dashboard(browser, url, dashboard, run, tag):
    page = browser.new_page(viewport={'width': 1700, 'height': 950},
                            timezone_id='America/New_York')
    page.goto(url + '/#' + dashboard, wait_until='networkidle')
    page.locator('input').nth(0).fill('^' + run + '$')
    page.locator('input').nth(1).fill(tag)
    page.locator('input').nth(1).press('Enter')
    page.wait_for_timeout(2500)
    return page


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:6006')
    parser.add_argument('--question', choices=['q2', 'q3'], required=True)
    args = parser.parse_args()
    figures = ROOT / 'report/figures'
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        if args.question == 'q2':
            for name, tag in [('q2_hist_conv', r'^resnet\.layer1\.1\.conv1\.weight/grad$'),
                              ('q2_hist_bn', r'^resnet\.layer4\.0\.bn2\.bias/grad$')]:
                page = open_dashboard(browser, args.url, 'histograms', 'q2_epoch_rerun', tag)
                page.locator('paper-icon-button[icon="fullscreen"]').first.click()
                page.wait_for_timeout(1000)
                native = ROOT.parent / 'runs/q2_epoch_revision/native_histograms'
                native.mkdir(exist_ok=True)
                page.locator('tf-histogram-loader').first.screenshot(path=str(native / (name + '.png')))
                page.close()
        else:
            page = open_dashboard(browser, args.url, 'images', 'fcos_inference', '^test_images$')
            page.get_by_text('Show actual image size', exact=True).click()
            page.wait_for_timeout(1000)
            # Keep the starter's complete ten-image grid unchanged. Capture the
            # first two images at readable size in the TensorBoard viewport.
            page.screenshot(path=str(figures / 'q3_inference.png'),
                            clip={'x': 350, 'y': 0, 'width': 1330, 'height': 700})
        browser.close()


if __name__ == '__main__':
    main()
