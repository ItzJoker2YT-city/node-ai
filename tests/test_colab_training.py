"""Run with: python -m unittest discover -s tests -p 'test_*.py'."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'colab'))
from train_colab import should_update_best


def quality(*scores):
    return {'passed': sum(scores), 'cases': [{'score': score} for score in scores]}


class CheckpointSelectionTests(unittest.TestCase):
    def eligible(self, loss, candidate, best_loss=2.0, best=None):
        baseline = quality(3, 4)
        return should_update_best(loss, candidate, best_loss, best or baseline, 2.0, baseline)

    def test_rejects_loss_regression_even_with_more_keyword_points(self):
        self.assertFalse(self.eligible(2.01, quality(5, 5)))

    def test_rejects_baseline_case_regression_even_with_higher_total(self):
        self.assertFalse(self.eligible(1.9, quality(2, 6)))

    def test_accepts_higher_quality_with_baseline_guards_met(self):
        self.assertTrue(self.eligible(1.99, quality(4, 4)))

    def test_tied_quality_requires_lower_loss(self):
        self.assertTrue(self.eligible(1.9, quality(3, 4)))
        self.assertFalse(self.eligible(2.0, quality(3, 4)))

    def test_does_not_replace_better_conversation_checkpoint_for_loss_only(self):
        self.assertFalse(self.eligible(1.8, quality(3, 4), 1.9, quality(4, 4)))

    def test_rejects_nonfinite_loss(self):
        for value in [float('nan'), float('inf'), -float('inf')]:
            self.assertFalse(self.eligible(value, quality(5, 5)))

    def test_notebook_cells_compile_and_use_four_hour_cpu_command(self):
        notebook = json.loads((ROOT / 'colab/Node_30_minute_conversation.ipynb').read_text())
        code = []
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                compile(source, f'cell {i}', 'exec')
                code.append(source)
        self.assertTrue(any('"--seconds", "14400", "--device", "cpu", "--batch-size", "2"' in source for source in code))


if __name__ == '__main__':
    unittest.main()
