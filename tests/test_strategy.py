import unittest
from yauber_backtester._strategy import Strategy


class StrategyTestCase(unittest.TestCase):
    def test_init(self):
        ctx = {
            'params': {'a': 10},
            'test': True
        }
        s = Strategy(**ctx)
        self.assertEqual(s.kwargs, ctx)
        self.assertEqual(s.params, {'a': 10})
        self.assertEqual(s.name, 'BaseStrategy')
        self.assertRaises(NotImplementedError, s.calculate, None)
        self.assertRaises(NotImplementedError, s.compose_portfolio, None, None, None)
        self.assertEqual(str(s), 'BaseStrategy')
        self.assertEqual(repr(s), "Strategy<BaseStrategy>")
        self.assertEqual(s.initialize(), None)


if __name__ == '__main__':
    unittest.main()
