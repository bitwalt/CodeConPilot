"""Make an order on Binance

Example:
    python3 -m copylot.crypto.order_binance order_binance --symbol=BTCUSDT --side=BUY --type=LIMIT --price=0.01 --quantity=0.1
"""

import argparse
import copylot
from copylot.binance_api import BinanceAPI
from copylot.order import Order


class Order(object):
    """An order on Binance

    Attributes:
        symbol (str): The coin
        side (str): BUY or SELL
        type (str): LIMIT, MARKET, STOP_LOSS, STOP_LOSS_LIMIT, TAKE_PROFIT, TAKE_PROFIT_LIMIT, or TRAILING_STOP
        price (float): The price of the order
        quantity (float): The quantity of the order
    """

    def __init__(self, symbol, side, type, price, quantity):
        self.symbol = symbol
        self.side = side
        self.type = type
        self.price = price
        self.quantity = quantity

    def __str__(self):
        return '{} {} {} {} {}'.format(self.symbol, self.side, self.type, self.price, self.quantity)

    

def main():
    parser = argparse.ArgumentParser(
        description='Make an order on Binance',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--symbol', required=True, help='The symbol of the coin')
    parser.add_argument('--side', required=True, help='BUY or SELL')
    parser.add_argument('--type', required=True, help='LIMIT, MARKET, STOP_LOSS, STOP_LOSS_LIMIT, TAKE_PROFIT, TAKE_PROFIT_LIMIT, or TRAILING_STOP')
    parser.add_argument('--price', required=True, help='The price of the order')
    parser.add_argument('--quantity', required=True, help='The quantity of the order')
    parser.add_argument('--api_key', required=True, help='The API key')
    parser.add_argument('--api_secret', required=True, help='The API secret')

    args = parser.parse_args()

    api = BinanceAPI(args.api_key, args.api_secret)
    order = Order(args.symbol, args.side, args.type, float(args.price), float(args.quantity))
    api.order(order)