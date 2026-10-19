"""Source-level kernel compatibility checks; these do not launch CUDA."""
import importlib
from pathlib import Path
import sys
import unittest


class KernelMetadata(unittest.TestCase):
    def test_vendored_sources_resolve_without_loading_torch(self):
        root = Path(__file__).resolve().parents[1]/'runtime'
        sys.path.insert(0, str(root))
        try:
            for component in ('canonical_mean_square', 'canonical_mean_square512', 'canonical_mean_square1024',
                              'canonical_sparse_batch', 'canonical_sparse512', 'canonical_sparse1024',
                              'canonical_rope_batch', 'canonical_rope512', 'canonical_rope1024'):
                with self.subTest(component=component):
                    metadata = importlib.import_module('ds41.'+component).source_metadata()
                    self.assertEqual(metadata['arch'], 'sm_120')
                    self.assertTrue(metadata['source_sha256'])
            self.assertNotIn('torch', sys.modules)
        finally:
            sys.path.remove(str(root))


if __name__ == '__main__':
    unittest.main()
