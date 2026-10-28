import base64
import unittest
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path

from PIL import Image
from streamlit.testing.v1 import AppTest


APP_ROOT = Path(__file__).resolve().parents[1]


class ImageSources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.sources.append(dict(attrs).get("src", ""))


class ImageRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = AppTest.from_file(str(APP_ROOT / "app.py"), default_timeout=60).run()

    def assert_valid_inline_image(self, source):
        header, encoded = source.split(",", 1)
        self.assertIn(header, {"data:image/png;base64", "data:image/jpeg;base64"})
        with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as image:
            expected_type = "image/png" if image.format == "PNG" else "image/jpeg"
            self.assertEqual(header, f"data:{expected_type};base64")
            self.assertGreater(image.width, 0)
            self.assertGreater(image.height, 0)
            image.verify()

    def test_native_images_and_heatmaps_bypass_media_endpoint(self):
        mode_selector = next(x for x in self.app.selectbox if x.label == "Sensitivity mode")
        for mode in mode_selector.options:
            with self.subTest(mode=mode):
                mode_selector = next(x for x in self.app.selectbox if x.label == "Sensitivity mode")
                mode_selector.set_value(mode).run()
                self.assertEqual(len(self.app.exception), 0)
                images = self.app.get("imgs")
                self.assertEqual(len(images), 4)
                for element in images:
                    for image in element.proto.imgs:
                        self.assert_valid_inline_image(image.url)

    def test_workflow_icons_and_reference_images_remain_embedded(self):
        self.app.toggle[0].set_value(True).run()
        self.assertEqual(len(self.app.exception), 0)
        parser = ImageSources()
        for element in self.app.markdown:
            parser.feed(element.value)
        self.assertGreater(len(parser.sources), 10)
        for source in parser.sources:
            self.assert_valid_inline_image(source)


if __name__ == "__main__":
    unittest.main()
