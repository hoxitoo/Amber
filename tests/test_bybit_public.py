import json
import unittest

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None

from amber.exchange.bybit_public import BybitPublicClient


def _mock_client(routes: dict[str, dict]) -> "httpx.Client":
    def handler(request: "httpx.Request") -> "httpx.Response":
        body = routes.get(request.url.path)
        if body is None:
            return httpx.Response(404, json={"retCode": 10001, "retMsg": "not found"})
        return httpx.Response(200, content=json.dumps(body))

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.bybit.test")


@unittest.skipIf(httpx is None, "httpx not installed")
class TestBybitPublicClient(unittest.TestCase):
    def test_get_klines_parses_and_sorts_oldest_first(self):
        # Bybit returns newest-first rows: [start, open, high, low, close, volume, turnover]
        routes = {
            "/v5/market/kline": {
                "retCode": 0,
                "retMsg": "OK",
                "result": {
                    "list": [
                        ["1700000060000", "101", "102", "100", "101.5", "10", "1015"],
                        ["1700000000000", "100", "101", "99", "100.5", "42", "4200"],
                    ]
                },
            }
        }
        client = BybitPublicClient(client=_mock_client(routes))
        candles = client.get_klines("BTCUSDT")
        self.assertEqual([c.ts for c in candles], [1700000000000, 1700000060000])
        self.assertEqual(candles[0].close, 100.5)
        self.assertEqual(candles[0].tf, "1m")

    def test_market_snapshot_from_tickers(self):
        routes = {
            "/v5/market/tickers": {
                "retCode": 0,
                "retMsg": "OK",
                "result": {
                    "list": [
                        {
                            "symbol": "BTCUSDT",
                            "lastPrice": "100.1",
                            "bid1Price": "100.0",
                            "ask1Price": "100.2",
                            "fundingRate": "0.0001",
                            "openInterest": "123456",
                        }
                    ]
                },
            }
        }
        client = BybitPublicClient(client=_mock_client(routes))
        ticker, oi, funding = client.get_market_snapshot("BTCUSDT")
        self.assertEqual(ticker.bid, 100.0)
        self.assertEqual(ticker.ask, 100.2)
        self.assertEqual(oi.oi, 123456.0)
        self.assertEqual(funding.funding, 0.0001)

    def test_api_error_raises(self):
        routes = {"/v5/market/kline": {"retCode": 10002, "retMsg": "rate limited", "result": {}}}
        client = BybitPublicClient(client=_mock_client(routes))
        with self.assertRaises(RuntimeError):
            client.get_klines("BTCUSDT")


if __name__ == "__main__":
    unittest.main()
