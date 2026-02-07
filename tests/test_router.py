import unittest

import src.models.cnn.mobilenet  # noqa: F401 (register models)
import src.models.cnn.resnet  # noqa: F401 (register models)
import src.models.vit.vit  # noqa: F401 (register models)
from src.models.registry import ModelRegistry
from src.routing.family_router import FamilyRouter


class TestFamilyRouter(unittest.TestCase):
    def setUp(self):
        self.router = FamilyRouter()

    def test_detect_resnet(self):
        model = ModelRegistry.create_model("resnet18")
        family, group = self.router.detect_family(model)
        self.assertEqual(family, "resnet")
        self.assertEqual(group, "cnn")

    def test_detect_mobilenet(self):
        model = ModelRegistry.create_model("mobilenetv3_large")
        family, group = self.router.detect_family(model)
        self.assertEqual(family, "mobilenet")
        self.assertEqual(group, "cnn")

    def test_detect_vit(self):
        model = ModelRegistry.create_model("vit_tiny")
        family, group = self.router.detect_family(model)
        self.assertEqual(family, "vit")
        self.assertEqual(group, "transformer")

    def test_complexity_level(self):
        model = ModelRegistry.create_model("resnet50")  # Max complexity assumed ~1.0
        level = self.router.get_complexity_level(model)
        self.assertLessEqual(level, 1.0)
        self.assertGreater(level, 0.8)

        model_small = ModelRegistry.create_model("resnet18")
        level_small = self.router.get_complexity_level(model_small)
        self.assertLess(level_small, level)


if __name__ == "__main__":
    unittest.main()
