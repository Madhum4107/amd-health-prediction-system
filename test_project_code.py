import unittest
import pandas as pd

from project_code import generate_synthetic_data, train_baseline, validate_input, predict_demo, FEATURES


class SyntheticDemoTests(unittest.TestCase):
    def test_data_reproducible_and_label_rule(self):
        a = generate_synthetic_data(100, 5)
        b = generate_synthetic_data(100, 5)
        pd.testing.assert_frame_equal(a, b)
        expected = ((a["age"] > 50).astype(int) + a["fever"] + a["tiredness"] + a["cough"] >= 2).astype(int)
        self.assertTrue((a["rule_label"] == expected).all())

    def test_training_returns_metrics(self):
        data = generate_synthetic_data(200)
        model, metrics = train_baseline(data)
        self.assertIn("accuracy", metrics)
        self.assertIn("balanced_accuracy", metrics)
        self.assertGreater(metrics["test_samples"], 0)
        prediction = predict_demo(model, validate_input(25, 0, 1, 0))
        self.assertIn(prediction, [0, 1])

    def test_input_validation(self):
        self.assertEqual(list(validate_input(25, 0, 1, 0).columns), FEATURES)
        with self.assertRaises(ValueError):
            validate_input(121, 0, 0, 0)
        with self.assertRaises(ValueError):
            validate_input(25, 2, 0, 0)

    def test_rejects_small_data_and_missing_columns(self):
        with self.assertRaises(ValueError):
            generate_synthetic_data(9)
        with self.assertRaises(ValueError):
            train_baseline(pd.DataFrame({"age": [1]}))


if __name__ == "__main__":
    unittest.main()
