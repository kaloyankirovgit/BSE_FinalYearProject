import sys
import os
import io
import numpy as np

# Suppress BSE's startup output when importing
_buf = io.StringIO()
sys.stdout, _buf = _buf, sys.stdout
from BSE import *
sys.stdout, _buf = _buf, sys.stdout


class InventoryTrader:
    """Mixin that adds cash/inventory tracking to any BSE trader."""

    def init_inventory(self):
        self.cash = 10000.0
        self.inventory = 0
        self.original_balance = self.balance
        self.mtm_profit_history = []

    def get_mtm_profit(self, current_price):
        return self.cash + (self.inventory * current_price)

    def bookkeep_inventory(self, time, trade, order, vrbs):
        self.blotter.append(trade)
        self.blotter = self.blotter[-self.blotter_length:]

        price = trade['price']

        if trade['party1'] == self.tid:
            if trade['type'] == 'Bid':
                self.cash -= price
                self.inventory += 1
            else:
                self.cash += price
                self.inventory -= 1
        elif trade['party2'] == self.tid:
            if trade['type'] == 'Bid':
                self.cash += price
                self.inventory -= 1
            else:
                self.cash -= price
                self.inventory += 1

        self.n_trades += 1

        if vrbs:
            print('TID=%s cash=%d inventory=%d' % (self.tid, self.cash, self.inventory))

        self.del_order(order)

        if hasattr(self, 'strats') and hasattr(self, 'active_strat'):
            if self.strats is not None:
                mtm = self.get_mtm_profit(price)
                profit = mtm - 10000
                self.strats[self.active_strat]['profit'] = profit
                totalprofit = self.strats[self.active_strat]['profit']
                birthtime = self.strats[self.active_strat]['start_t']
                self.strats[self.active_strat]['pps'] = self.profitpertime_update(
                    time, birthtime, totalprofit)


class InventoryExchange(Exchange):

    def __init__(self):
        super().__init__()
        self.last_trade_price = 100
        self.qid = 0

    def get_mid_price(self):
        if self.bids.best_price is not None and self.asks.best_price is not None:
            return (self.bids.best_price + self.asks.best_price) / 2.0
        return self.last_trade_price

    def process_order2(self, time, order, tape_file, vrbs):
        verbose = False

        if order.otype == 'Bid':
            if self.asks.n_orders > 0 and order.price >= self.asks.best_price:
                counterparty_tid = self.asks.delete_best()
                traded_price = self.asks.best_price if self.asks.best_price is not None else order.price

                record = {
                    'type': 'Bid', 'time': time, 'price': traded_price, 'qty': 1,
                    'party1': order.tid, 'party2': counterparty_tid,
                }

                self.last_trade_price = traded_price
                self.tape.append(record)
                self.tape = self.tape[-self.tape_length:]

                if tape_file is not None:
                    tape_file.write('Trade, %s, %s, %s, %s, %s\n' % (
                        record['time'], record['price'],
                        record['party1'], record['party2'], record['qty']))

                if verbose:
                    print('>>>>>>>>>>>>>>>>>TRADE t=%5.2f $%d %s %s' %
                          (time, traded_price, counterparty_tid, order.tid))

                return record

        elif order.otype == 'Ask':
            if self.bids.n_orders > 0 and order.price <= self.bids.best_price:
                counterparty_tid = self.bids.delete_best()
                traded_price = self.bids.best_price if self.bids.best_price is not None else order.price

                record = {
                    'type': 'Ask', 'time': time, 'price': traded_price, 'qty': 1,
                    'party1': order.tid, 'party2': counterparty_tid,
                }

                self.last_trade_price = traded_price
                self.tape.append(record)
                self.tape = self.tape[-self.tape_length:]

                if tape_file is not None:
                    tape_file.write('Trade, %s, %s, %s, %s, %s\n' % (
                        record['time'], record['price'],
                        record['party1'], record['party2'], record['qty']))

                if verbose:
                    print('>>>>>>>>>>>>>>>>>TRADE t=%5.2f $%d %s %s' %
                          (time, traded_price, order.tid, counterparty_tid))

                return record

        return None


def trader_type_inventory(robottype, name, balance, params, time):
    """Factory that returns an inventory-aware trader of the given type."""
    types = {
        'GVWY':   InventoryGVWY,
        'ZIC':    InventoryZIC,
        'SHVR':   InventorySHVR,
        'SNPR':   InventorySNPR,
        'ZIP':    InventoryZIP,
        'PRSH':   InventoryPRSH,
        'CUSTOM': InventoryCUSTOM,
    }
    if robottype not in types:
        sys.exit('FATAL: unknown trader type %s\n' % robottype)
    return types[robottype](robottype, name, balance, params, time)


class InventoryGVWY(InventoryTrader, TraderGiveaway):
    def __init__(self, ttype, tid, balance, params, time):
        TraderGiveaway.__init__(self, ttype, tid, balance, params, time)
        self.init_inventory()


class InventoryZIC(InventoryTrader, TraderZIC):
    def __init__(self, ttype, tid, balance, params, time):
        TraderZIC.__init__(self, ttype, tid, balance, params, time)
        self.init_inventory()


class InventorySHVR(InventoryTrader, TraderShaver):
    def __init__(self, ttype, tid, balance, params, time):
        TraderShaver.__init__(self, ttype, tid, balance, params, time)
        self.init_inventory()


class InventorySNPR(InventoryTrader, TraderSniper):
    def __init__(self, ttype, tid, balance, params, time):
        TraderSniper.__init__(self, ttype, tid, balance, params, time)
        self.init_inventory()


class InventoryZIP(InventoryTrader, TraderZIP):
    def __init__(self, ttype, tid, balance, params, time):
        TraderZIP.__init__(self, ttype, tid, balance, params, time)
        self.init_inventory()


class InventoryPRSH(InventoryTrader, TraderPRZI):
    def __init__(self, ttype, tid, balance, params, time):
        if 'strat_wait_time' in params:
            params['optimizer'] = 'PRSH'
            strat_wait = params['strat_wait_time']
        else:
            strat_wait = 7200
        TraderPRZI.__init__(self, ttype, tid, balance, params, time)
        self.strat_wait_time = strat_wait
        self.strat_eval_time = self.k * self.strat_wait_time
        self.init_inventory()


class InventoryCUSTOM(InventoryTrader, TraderCustom):
    """
    Inventory-aware wrapper for TraderCustom.
    Overrides volatility calculation and defensive mode logic to work correctly
    under the mark-to-market profit framework used in this study.
    """

    def __init__(self, ttype, tid, balance, params, time):
        TraderCustom.__init__(self, ttype, tid, balance, params, time)
        self.init_inventory()
        self.bookkeep = self.bookkeep_inventory

    def _update_defensive_mode(self, volatility):
        abs_inv = abs(self.inventory)
        if volatility > self.volatility_threshold or abs_inv > self.position_limit:
            self.defensive_mode = True
        elif volatility < self.volatility_threshold * 0.5 and abs_inv < self.position_limit * 0.5:
            self.defensive_mode = False

    def _calculate_volatility(self):
        # Volatility measured as std of consecutive price changes rather than
        # raw price levels, which would always be high given the wide limit-price
        # distribution and make the threshold parameter meaningless.
        if len(self.price_history) < 10:
            return 0.0
        recent = self.price_history[-min(50, len(self.price_history)):]
        if len(recent) < 2:
            return 0.0
        changes = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
        return float(np.std(changes))

    def getorder(self, time, countdown, lob):
        if len(self.orders) < 1:
            return None

        volatility = self._calculate_volatility()
        self._update_defensive_mode(volatility)

        otype = self.orders[0].otype
        limit_price = self.orders[0].price
        best_bid = lob['bids']['best'] if lob['bids']['n'] > 0 else None
        best_ask = lob['asks']['best'] if lob['asks']['n'] > 0 else None
        qid = lob['QID']

        if self.defensive_mode and self.inventory != 0:
            reduces = (self.inventory > 0 and otype == 'Ask') or \
                      (self.inventory < 0 and otype == 'Bid')
            if reduces:
                if otype == 'Ask' and best_bid is not None:
                    qprice = max(best_bid, int(limit_price * 0.99))
                elif otype == 'Bid' and best_ask is not None:
                    qprice = min(best_ask, int(limit_price * 1.01))
                else:
                    qprice = limit_price
                qprice = max(bse_sys_minprice, min(bse_sys_maxprice, qprice))
                order = Order(self.tid, otype, qprice, self.orders[0].qty, time, qid)
                self.lastquote = order
                return order

        return TraderCustom.getorder(self, time, countdown, lob)
