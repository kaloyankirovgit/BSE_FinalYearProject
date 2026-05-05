"""
Runs the final 150-trial evaluation using the optimal Custom Trader parameters
identified by run_simulation.py. Results are written to data/final_results.csv.
Run this after run_simulation.py has completed and data/optimal_parameters.csv exists.
"""

import sys
import os
import io
import csv
import random
import contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from inventory_wrapper import *


# Output paths
DATA_DIR        = os.path.join(HERE, 'data')
RESULTS_OUT     = os.path.join(DATA_DIR, 'final_results.csv')
CHECKPOINT      = os.path.join(DATA_DIR, 'final_checkpoint.csv')
PARAMS_FILE     = os.path.join(DATA_DIR, 'optimal_parameters.csv')

# Simulation settings
N_TRIALS         = 150
SESSION_DURATION = 3600
CONDITIONS       = ['stable', 'shock_up', 'shock_down', 'multi_regime']
TRADER_TYPES     = ['GVWY', 'ZIC', 'SHVR', 'SNPR', 'ZIP', 'PRSH', 'CUSTOM']
N_PER_TYPE       = 5
TRACKER_INTERVAL = 10
INIT_CASH        = 10000

PRSH_PARAMS = {'k': 4, 'strat_min': -1.0, 'strat_max': 1.0, 'strat_wait_time': 60}


@contextlib.contextmanager
def _quiet():
    """Suppress BSE's verbose output during simulation steps."""
    old, sys.stdout = sys.stdout, io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def load_optimal_params():
    if not os.path.exists(PARAMS_FILE):
        sys.exit('ERROR: %s not found. Run run_simulation.py first.' % PARAMS_FILE)
    row = pd.read_csv(PARAMS_FILE).iloc[0]
    params = {
        'alpha': float(row['alpha']),
        'position_limit': int(row['position_limit']),
        'volatility_threshold': float(row['volatility_threshold']),
        'initial_margin': 0.05,
        'defensive_margin': 0.15,
    }
    print('Loaded optimal parameters from %s:' % PARAMS_FILE)
    print('  alpha                = %.6f' % params['alpha'])
    print('  position_limit       = %d'   % params['position_limit'])
    print('  volatility_threshold = %.6f' % params['volatility_threshold'])
    return params


def get_offset(t, condition):
    if condition == 'stable':
        return 0
    elif condition == 'shock_up':
        return 30 if t >= SESSION_DURATION / 3 else 0
    elif condition == 'shock_down':
        return -30 if t >= SESSION_DURATION / 3 else 0
    elif condition == 'multi_regime':
        regime = int(t / (SESSION_DURATION / 10))
        if regime == 0:
            return 0
        m = np.random.normal(1.0, 0.2)
        m = max(0.5, min(1.5, m))
        return (m - 1.0) * 100
    return 0


def build_traders(custom_params):
    traders = {}
    tc = 0
    for ttype in TRADER_TYPES:
        prefix = 'C' if ttype == 'CUSTOM' else ttype[0]
        for _ in range(N_PER_TYPE):
            name = '%s%02d' % (prefix, tc)
            if ttype == 'PRSH':
                params = PRSH_PARAMS.copy()
            elif ttype == 'CUSTOM':
                p = custom_params.copy()
                p['beta'] = p.pop('alpha', p.get('beta', 0.05))
                params = p
            else:
                params = None
            with _quiet():
                traders[name] = trader_type_inventory(ttype, name, 0, params, 0)
            tc += 1
    return traders


def run_single_trial(condition, trial_num, custom_params):
    traders = build_traders(custom_params)
    exchange = InventoryExchange()
    base_price, rng = 100, 20

    price_history = []
    inv_snapshots = []

    for time in range(SESSION_DURATION):
        offset = get_offset(time, condition)
        trader_list = list(traders.values())
        random.shuffle(trader_list)

        for trader in trader_list:
            otype = 'Bid' if random.random() < 0.5 else 'Ask'
            if otype == 'Bid':
                lp = int(base_price + offset + random.randint(0, rng))
            else:
                lp = int(base_price + offset + random.randint(-rng, 0))
            lp = max(bse_sys_minprice, min(bse_sys_maxprice, lp))
            order = Order(trader.tid, otype, lp, 1, time, None)
            trader.add_order(order, False)

            countdown = (SESSION_DURATION - time) / SESSION_DURATION
            with _quiet():
                lob = exchange.publish_lob(time, None, False)
                quote = trader.getorder(time, countdown, lob)

            if quote is not None:
                exchange.add_order(quote, False)
                with _quiet():
                    trade = exchange.process_order2(time, quote, None, False)
                if trade is not None:
                    price_history.append(trade['price'])
                    p1, p2 = trade['party1'], trade['party2']
                    for tr in traders.values():
                        if tr.tid in (p1, p2):
                            tr.bookkeep_inventory(time, trade, None, False)

        last_trade = exchange.tape[-1] if exchange.tape else None
        with _quiet():
            lob_r = exchange.publish_lob(time, None, False)
        for trader in traders.values():
            with _quiet():
                trader.respond(time, lob_r, last_trade, False)

        if time % TRACKER_INTERVAL == 0:
            by_type = defaultdict(list)
            for trader in traders.values():
                by_type[trader.ttype].append(trader.inventory)
            for ttype, invs in by_type.items():
                inv_snapshots.append({
                    'trial': trial_num, 'condition': condition,
                    'timestep': time // TRACKER_INTERVAL,
                    'ttype': ttype, 'mean_inv': float(np.mean(invs)),
                })

    final_mid = float(np.mean(price_history[-20:])) if price_history else exchange.get_mid_price()

    results = []
    for trader in traders.values():
        mtm = trader.get_mtm_profit(final_mid) - INIT_CASH
        results.append({
            'trial': trial_num, 'condition': condition,
            'tid': trader.tid, 'ttype': trader.ttype,
            'final_profit': round(mtm, 2),
            'n_trades': trader.n_trades,
            'final_inventory': trader.inventory,
        })

    return results, inv_snapshots


def append_rows(filepath, rows, fieldnames):
    write_header = not os.path.exists(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def completed_trials(checkpoint_file):
    done = set()
    if not os.path.exists(checkpoint_file):
        return done
    with open(checkpoint_file) as f:
        for row in csv.DictReader(f):
            done.add((row['condition'], int(row['trial'])))
    return done


def run_evaluation(custom_params):
    os.makedirs(DATA_DIR, exist_ok=True)

    results_fields    = ['trial', 'condition', 'tid', 'ttype',
                         'final_profit', 'n_trades', 'final_inventory']
    inv_fields        = ['trial', 'condition', 'timestep', 'ttype', 'mean_inv']
    checkpoint_fields = ['condition', 'trial', 'timestamp']

    done      = completed_trials(CHECKPOINT)
    total     = len(CONDITIONS) * N_TRIALS
    completed = len(done)

    print('\nFinal Evaluation')
    print('Traders: %s (%d each, %d total per session)' % (
        ', '.join(TRADER_TYPES), N_PER_TYPE, N_PER_TYPE * len(TRADER_TYPES)))
    print('Trials per condition: %d  |  Total: %d  |  Already done: %d\n' % (
        N_TRIALS, total, completed))

    trial_times = []

    for condition in CONDITIONS:
        print('Condition: %s' % condition)
        for trial in range(N_TRIALS):
            if (condition, trial) in done:
                continue
            t0 = datetime.now()
            results, inv_snaps = run_single_trial(condition, trial, custom_params)
            append_rows(RESULTS_OUT,  results,    results_fields)
            append_rows(CHECKPOINT,
                        [{'condition': condition, 'trial': trial,
                          'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}],
                        checkpoint_fields)

            elapsed = (datetime.now() - t0).total_seconds()
            trial_times.append(elapsed)
            completed += 1
            remaining = total - completed
            avg_t = float(np.mean(trial_times[-20:]))
            eta_s = int(avg_t * remaining)
            eta_str = '%dh %dm' % (eta_s // 3600, (eta_s % 3600) // 60)

            if trial % 25 == 0:
                custom_p = [r['final_profit'] for r in results if r['ttype'] == 'CUSTOM']
                c_mean = np.mean(custom_p) if custom_p else 0
                print('  [%04d/%d] %s trial %03d | %.1fs | CUSTOM mean: %+.0f | ETA: %s' % (
                    completed, total, condition, trial, elapsed, c_mean, eta_str))

    print('\nEvaluation complete. Results written to data/final_results.csv')

    df = pd.read_csv(RESULTS_OUT)
    summary = (df.groupby('ttype').agg(
        mean_profit=('final_profit', 'mean'),
        std_profit=('final_profit', 'std'),
        mean_trades=('n_trades', 'mean'),
        pct_positive=('final_profit', lambda x: (x > 0).mean() * 100),
    ).sort_values('mean_profit', ascending=False).round(2))
    summary.to_csv(os.path.join(DATA_DIR, 'summary_statistics.csv'))

    print('\nResults summary:')
    print('  %-8s  %18s  %10s  %12s  %13s' % (
        'Trader', 'Mean Profit (ECU)', 'Std', 'Mean Trades', '% Profitable'))
    for t, row in summary.iterrows():
        print('  %-8s  %+18.2f  %10.2f  %12.1f  %12.1f%%' % (
            t, row['mean_profit'], row['std_profit'],
            row['mean_trades'], row['pct_positive']))


if __name__ == '__main__':
    print('BSE Final Evaluation')
    print('Started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    params = load_optimal_params()
    run_evaluation(params)
    print('\nDone.')
