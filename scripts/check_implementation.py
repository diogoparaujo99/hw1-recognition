"""Focused checks beyond the provided detector tests."""
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'q1_q2_classification'))
from trainer import masked_binary_cross_entropy
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
# Load detection modules after the classification imports to avoid utils clashes.
sys.path.insert(0, str(ROOT / 'detection'))
sys.modules.pop('utils', None)
from detection_helper import inference_with_detector
from detection_utils import get_fpn_location_coords, fcos_make_centerness_targets
from one_stage_detector import FCOS, FCOSPredictionNetwork


class Checks(unittest.TestCase):
    def test_bce_and_gradients(self):
        x = torch.tensor([[-1000., -1., 0., 1., 1000.]], requires_grad=True)
        target = torch.tensor([[1., 0., 1., 1., 0.]])
        weights = torch.tensor([[1., 1., 1., 0., 1.]])
        actual = masked_binary_cross_entropy(x, target, weights)
        reference = (torch.nn.functional.binary_cross_entropy_with_logits(
            x, target, reduction='none') * weights).sum() / weights.sum()
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(torch.autograd.grad(actual, x, retain_graph=True)[0],
                                   torch.autograd.grad(reference, x)[0])
        self.assertEqual(masked_binary_cross_entropy(x, target, weights * 0).item(), 0)

    def test_annotation_mask_and_eval_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'ImageSets/Main').mkdir(parents=True)
            (root / 'Annotations').mkdir()
            (root / 'ImageSets/Main/test.txt').write_text('sample\n')
            (root / 'Annotations/sample.xml').write_text('''<annotation>
                <object><name>dog</name><difficult>0</difficult></object>
                <object><name>dog</name><difficult>1</difficult></object>
                <object><name>person</name><difficult>0</difficult></object></annotation>''')
            dataset = VOCDataset('test', 64, tmp)
            labels, weights = dataset.anno_list[0]
            self.assertEqual(labels.sum().item(), 2)
            self.assertEqual(weights[VOCDataset.get_class_index('dog')].item(), 0)
            self.assertFalse(dataset.augment)

    def test_shapes_and_rectangular_coordinates(self):
        self.assertEqual(SimpleCNN(20, 64, 3)(torch.randn(2, 3, 64, 64)).shape, (2, 20))
        coords = get_fpn_location_coords({'p3': (1, 64, 2, 3)}, {'p3': 8})['p3']
        torch.testing.assert_close(coords, torch.tensor([[4.,4.],[12.,4.],[20.,4.],
                                                       [4.,12.],[12.,12.],[20.,12.]]))
        net = FCOSPredictionNetwork(20, 64, [32])
        self.assertEqual(net({'p3': torch.randn(2,64,2,3)})[0]['p3'].shape, (2,6,20))

    def test_no_predictions_still_writes_ground_truth(self):
        class EmptyDetector(nn.Module):
            def forward(self, images, **kwargs):
                return torch.empty(0,4), torch.empty(0,dtype=torch.long), torch.empty(0)
        batch = (['sample.jpg'], torch.zeros(1,3,8,8), torch.tensor([[[1.,1.,7.,7.,0.]]]))
        with tempfile.TemporaryDirectory() as tmp:
            inference_with_detector(EmptyDetector(), [batch], {0:'dog'}, .4,.6, output_dir=tmp)
            self.assertTrue((Path(tmp)/'ground-truth/sample.txt').read_text().startswith('dog '))
            self.assertEqual((Path(tmp)/'detection-results/sample.txt').read_text(), '')

    def test_background_only_loss_and_empty_inference(self):
        model = FCOS(20, 64, [64,64])
        images = torch.randn(2,3,64,64)
        loss = sum(model(images, torch.full((2,40,5),-1.)).values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        model.eval()
        boxes, classes, scores = model(images[:1], test_score_thresh=1.1)
        self.assertEqual(boxes.shape, (0,4))
        self.assertEqual(classes.numel(), 0)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
