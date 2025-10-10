import logging
import unittest

from copick_torch.logging import setup_logging


class TestLogging(unittest.TestCase):
    def test_setup_logging(self):
        # Test that setup_logging returns a logger
        logger = setup_logging()
        self.assertIsInstance(logger, logging.Logger)

        # Test that logger name is set correctly
        self.assertEqual(logger.name, "copick_torch")

        # Test that logger level is INFO
        self.assertEqual(logger.level, logging.INFO)

        # Test that the logger has a handler for stdout
        self.assertTrue(any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers))


if __name__ == "__main__":
    unittest.main()
