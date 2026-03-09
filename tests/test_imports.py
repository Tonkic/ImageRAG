import unittest
import sys
import os
from unittest.mock import MagicMock

class MockModule(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

class sys_modules_dict(dict):
    def __missing__(self, key):
        if key in ['torch', 'torchvision', 'transformers', 'diffusers', 'PIL', 'openai', 'qwen_vl_utils', 'psutil', 'rank_bm25', 'PIL.PngImagePlugin', 'PIL.Image']:
            return MockModule()
        raise KeyError(key)

sys.modules = sys_modules_dict(sys.modules)

# Add the project root to sys.path so that 'src' is resolvable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

class TestImports(unittest.TestCase):
    def test_utils_imports(self):
        try:
            import src.utils.rag_utils
            import src.utils.utils
        except ImportError as e:
            self.fail(f"Utils imports failed: {e}")

    def test_critical_imports(self):
        try:
            import src.critical.taxonomy_aware_critic
            import src.critical.binary_critic
        except ImportError as e:
            self.fail(f"Critical module imports failed: {e}")

if __name__ == "__main__":
    unittest.main()
