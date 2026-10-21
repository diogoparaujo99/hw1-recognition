"""Capture TensorBoard's own histogram and inference displays."""
import argparse
import io
from pathlib import Path
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

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
            # Reuse the first six renderings from the starter's eight-column
            # grid. Only their layout changes; boxes and predictions stay intact.
            logs = ROOT.parent / 'runs/starter_revision'
            events = EventAccumulator(str(logs / 'fcos_inference'),
                                      size_guidance={'images': 0}).Reload()
            source = events.Images('test_images')[-1]
            original = Image.open(io.BytesIO(source.encoded_image_string)).convert('RGB')
            # Original grid: eight columns, two rows, two-pixel padding.
            width = (original.width - 18) // 8
            height = (original.height - 6) // 2
            assert original.size == (8 * width + 18, 2 * height + 6)
            grid = Image.new('RGB', (2 * width + 6, 3 * height + 8))
            for index in range(6):
                left = 2 + index * (width + 2)
                tile = original.crop((left, 2, left + width, 2 + height))
                grid.paste(tile, (2 + index % 2 * (width + 2),
                                 2 + index // 2 * (height + 2)))
            with SummaryWriter(str(logs / 'fcos_inference_six')) as writer:
                writer.add_image('test_images', np.asarray(grid), source.step, dataformats='HWC')
            page = open_dashboard(browser, args.url, 'images', 'fcos_inference_six', '^test_images$')
            page.set_viewport_size({'width': 1700, 'height': 1900})
            page.get_by_text('Show actual image size', exact=True).click()
            page.wait_for_timeout(1000)
            image = page.locator('img').first
            image.scroll_into_view_if_needed()
            bounds = image.bounding_box()
            # Include the complete grid and TensorBoard's run/step header.
            page.screenshot(path=str(figures / 'q3_inference.png'),
                            clip={'x': bounds['x'] - 2, 'y': bounds['y'] - 36,
                                  'width': bounds['width'] + 4, 'height': bounds['height'] + 38})
        browser.close()


if __name__ == '__main__':
    main()
