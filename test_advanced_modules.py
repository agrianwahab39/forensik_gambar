import unittest
import numpy as np
from PIL import Image

import advanced_analysis as aa
import classification
import ela_analysis

class TestAdvancedModules(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (64, 64), color='white')
        self.analysis_results = {
            'ela_mean': 5.0,
            'ela_std': 2.0,
            'ela_regional_stats': {
                'regional_inconsistency': 0.1,
                'outlier_regions': 1,
                'suspicious_regions': [],
                'mean_variance': 0.0,
                'std_variance': 0.0
            },
            'sift_matches': 0,
            'ransac_inliers': 0,
            'geometric_transform': None,
            'block_matches': [],
            'noise_analysis': {'overall_inconsistency': 0.1},
            'jpeg_ghost_suspicious_ratio': 0.05,
            'jpeg_analysis': {
                'response_variance': 0.0,
                'double_compression_indicator': 0.0,
                'estimated_original_quality': 80,
                'compression_inconsistency': False
            },
            'frequency_analysis': {
                'frequency_inconsistency': 0.5,
                'dct_stats': {'freq_ratio': 0.1}
            },
            'texture_analysis': {'overall_inconsistency': 0.2},
            'edge_analysis': {'edge_inconsistency': 0.2},
            'illumination_analysis': {'overall_illumination_inconsistency': 0.2},
            'statistical_analysis': {
                'R_entropy': 5,
                'G_entropy': 5,
                'B_entropy': 5,
                'rg_correlation': 0.9,
                'rb_correlation': 0.9,
                'gb_correlation': 0.9,
                'overall_entropy': 7
            },
            'metadata': {
                'Metadata_Inconsistency': [],
                'Metadata_Authenticity_Score': 80,
            },
            'localization_analysis': {'tampering_percentage': 0, 'kmeans_localization': {'cluster_ela_means': []}},
            'jpeg_ghost': np.zeros((64,64)),
            'noise_map': np.zeros((64,64)),
            'classification': {}
        }

    def test_advanced_analysis_functions(self):
        noise = aa.analyze_noise_consistency(self.image)
        freq = aa.analyze_frequency_domain(self.image)
        tex = aa.analyze_texture_consistency(self.image)
        edge = aa.analyze_edge_consistency(self.image)
        illum = aa.analyze_illumination_consistency(self.image)
        stats = aa.perform_statistical_analysis(self.image)
        self.assertIn('overall_inconsistency', noise)
        self.assertIn('frequency_inconsistency', freq)
        self.assertIn('overall_inconsistency', tex)
        self.assertIn('edge_inconsistency', edge)
        self.assertIn('overall_illumination_inconsistency', illum)
        self.assertIn('overall_entropy', stats)

    def test_classification(self):
        result = classification.classify_manipulation_advanced(self.analysis_results)
        self.assertIn('type', result)
        self.assertIn('confidence', result)
        fv = classification.prepare_feature_vector(self.analysis_results)
        self.assertTrue(isinstance(fv, np.ndarray))

    def test_ela(self):
        # Use multiple qualities to match the default weight configuration
        ela_result = ela_analysis.perform_multi_quality_ela(self.image)
        self.assertEqual(len(ela_result), 6)

if __name__ == '__main__':
    unittest.main()
