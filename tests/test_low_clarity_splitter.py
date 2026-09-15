import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from low_clarity_splitter import split_low_clarity_instances


def fixture():
    labels = np.zeros((260, 200), np.uint16)
    for index, x in enumerate((25, 65, 105, 132), 1):
        ellipse = np.zeros(labels.shape, np.uint8)
        cv2.ellipse(ellipse, (x,130), (12,105), 0, 0,360,1,-1)
        labels[ellipse > 0] = index
    probability = np.zeros((3, *labels.shape),np.float32)
    foreground = labels > 0
    probability[0] = np.where(foreground,.01,.98)
    probability[1] = np.where(foreground,.97,.01)
    probability[2] = 1-probability[0]-probability[1]
    image = np.where(foreground,.70,.02).astype(np.float32)
    # An uncertain bridge creates one label while the two tips remain separate.
    merged = labels.copy()
    merged[merged==4] = 3
    merged[95:165,115:123] = 3
    probability[:,95:165,115:123] = np.asarray([.05,.35,.60])[:,None,None]
    image[95:165,115:123] = .15
    return labels,merged,probability,image


class SplitterTests(unittest.TestCase):
    def test_two_heads_split_and_union_preserved(self):
        _,merged,p,image=fixture()
        result,reports=split_low_clarity_instances(merged,p,image)
        self.assertEqual(len(reports),1)
        self.assertEqual(int(result.max()),4)
        self.assertTrue(np.array_equal(result>0,merged>0))
        self.assertFalse(np.any((result==3)&(result==4)))
        self.assertEqual(reports[0].status,'REVIEW_LOW_CLARITY_SPLIT')
        self.assertEqual(reports[0].resulting_labels,'3|4')

    def test_no_head_gap_no_split(self):
        _,merged,p,image=fixture()
        # Bright material spans both heads: long low-interior stripe alone is
        # insufficient evidence for changing the instance count.
        image[:100,85:150] = .70
        result,reports=split_low_clarity_instances(merged,p,image)
        self.assertEqual(reports,[])
        self.assertTrue(np.array_equal(result,merged))

    def test_correct_separation_is_unchanged(self):
        labels,_,p,image=fixture()
        result,reports=split_low_clarity_instances(labels,p,image)
        self.assertEqual(reports,[])
        self.assertTrue(np.array_equal(result,labels))

    def test_shortened_core_does_not_truncate_tail(self):
        _,merged,p,image=fixture()
        # One confident core stops early even though its foreground continues.
        p[:,175:220,124:145]=np.asarray([.02,.40,.58])[:,None,None]
        result,reports=split_low_clarity_instances(merged,p,image)
        self.assertEqual(len(reports),1)
        for label in (3,4):
            self.assertGreater(np.nonzero(result==label)[0].max(),220)
        self.assertTrue(np.array_equal(result>0,merged>0))

    def test_safety_cap_does_not_force_splitting(self):
        _,merged,p,image=fixture()
        result,reports=split_low_clarity_instances(merged,p,image,max_instances=3)
        self.assertEqual(reports,[])
        self.assertTrue(np.array_equal(result,merged))

    def test_empty_and_invalid_evidence(self):
        labels,_,p,image=fixture()
        result,reports=split_low_clarity_instances(np.zeros_like(labels),p,image)
        self.assertFalse(result.any())
        self.assertEqual(reports,[])
        with self.assertRaises(ValueError):
            split_low_clarity_instances(labels,p[:,:-1],image)
        p[0,0,0]=np.nan
        with self.assertRaises(ValueError):
            split_low_clarity_instances(labels,p,image)


if __name__=='__main__':
    unittest.main()
