import unittest

from vv4ml.utils.model_factory import MODEL_REGISTRY, SEED_PARAM, recreate_model


class TestModelFactory(unittest.TestCase):
    """Test class for model_factory module"""

    def test_recreate_model_sets_seed(self):
        """Every model with a seed parameter receives the requested random_state"""
        for model_type, seed_param in SEED_PARAM.items():
            with self.subTest(model_type=model_type):
                model = recreate_model(model_type, {}, random_state=7)

                self.assertEqual(model.get_params()[seed_param], 7)

    def test_recreate_model_sets_seed_gpr(self):
        """The GPR pipeline receives the requested random_state"""
        model = recreate_model("GaussianProcessRegressor", {}, random_state=7)

        self.assertEqual(model.get_params()["gpr__random_state"], 7)

    def test_recreate_model_without_seed(self):
        """Models without a seed parameter are still instantiated"""
        seedless = ["SupportVectorRegressor", "LinearRegressor", "PLSRegression"]

        for model_type in seedless:
            with self.subTest(model_type=model_type):
                model = recreate_model(model_type, {}, random_state=7)

                self.assertIsInstance(model, MODEL_REGISTRY[model_type])

    def test_seed_param_keys_are_registered(self):
        """SEED_PARAM only refers to models in MODEL_REGISTRY"""
        self.assertLessEqual(set(SEED_PARAM), set(MODEL_REGISTRY))

    def test_recreate_model_unknown_type(self):
        """An unknown model type raises ValueError"""
        with self.assertRaises(ValueError):
            recreate_model("NotAModel", {})
