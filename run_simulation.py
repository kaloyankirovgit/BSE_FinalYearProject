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

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args


# Output paths
DATA_DIR    = os.path.join(HERE, 'data')
BAYES_LOG   = os.path.join(DATA_DIR, 'bayesian_search_log.csv')
BAYES_OPT   = os.path.join(DATA_DIR, 'optimal_parameters.csv')
RESULTS_OUT = os.path.join(DATA_DIR, 'simulation_results.csv')
INV_OUT     = os.path.join(DATA_DIR, 'inventory_timeseries.csv')
CHECKPOINT  = os.path.join(DATA_DIR, 'simulation_checkpoint.csv')

# Simulation settings
N_TRIALS         = 150
N_TRIALS_OPT     = 10
SESSION_DURATION = 3600
CONDITIONS       = ['stable', 'shock_up', 'shock_down', 'multi_regime']
TRADER_TYPES     = ['GVWY', 'ZIC', 'SHVR', 'SNPR', 'ZIP', 'PRSH', 'CUSTOM']
N_PER_TYPE       = 5
TRACKER_INTERVAL = 10
INIT_CASH        = 10000

PRSH_PARAMS = {'k': 4, 'strat_min': -1.0, 'strat_max': 1.0, 'strat_wait_time': 60}

# Starting values for alpha; replaced by Bayesian optimisation before the main run
CUSTOM_PARAMS = {
    'alpha': 0.01,
    'position_limit': 400,
    'volatility_threshold': 10.787,
    'initial_margin': 0.05,
    'defensive_margin': 0.15,
}


@contextlib.contextmanager
def _quiet():
    """Suppress BSE's verbose output during simulation steps."""
    old, sys.stdout = sys.stdout, io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


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


def build_traders(custom_params=None):
    if custom_params is None:
        custom_params = CUSTOM_PARAMS
    traders = {}
    tc = 0
    for ttype in TRADER_TYPES:
        prefix = 'C' if ttype == 'CUSTOM' else ttype[0]
        for _ in range(N_PER_TYPE):
            name = '%s%02d' % (prefix, tc)
            if ttype == 'PRSH':
                params = PRSH_PARAMS.copy()
            elif ttype == 'CUSTOM':
                # Map alpha to beta so BSE.py's TraderCustom receives the right key
                p = custom_params.copy()
                p['beta'] = p.pop('alpha', p.get('beta', 0.05))
                params = p
            else:
                params = None
            with _quiet():
                traders[name] = trader_type_inventory(ttype, name, 0, params, 0)
            tc += 1
    return traders


def run_single_trial(condition, trial_num, custom_params=None):
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


# Bayesian optimisation

_bo_log = []
_bo_start = None

_search_space = [
    Real(0.001, 0.30, name='alpha'),
    Integer(20, 800,  name='position_limit'),
    Real(3.0, 15.0,   name='volatility_threshold'),
]


def _bo_objective_simulation(alpha, position_limit, volatility_threshold):
    params = {
        'alpha': float(alpha),
        'position_limit': int(position_limit),
        'volatility_threshold': float(volatility_threshold),
        'initial_margin': 0.05,
        'defensive_margin': 0.15,
    }
    profits = []
    for condition in CONDITIONS:
        for _ in range(N_TRIALS_OPT):
            traders = build_traders(custom_params=params)
            exchange = InventoryExchange()
            for time in range(SESSION_DURATION):
                offset = get_offset(time, condition)
                trader_list = list(traders.values())
                random.shuffle(trader_list)
                for trader in trader_list:
                    otype = 'Bid' if random.random() < 0.5 else 'Ask'
                    if otype == 'Bid':
                        lp = int(100 + offset + random.randint(0, 20))
                    else:
                        lp = int(100 + offset + random.randint(-20, 0))
                    lp = max(bse_sys_minprice, min(bse_sys_maxprice, lp))
                    order = Order(trader.tid, otype, lp, 1, time, None)
                    trader.add_order(order, False)
                    with _quiet():
                        lob = exchange.publish_lob(time, None, False)
                        quote = trader.getorder(time, (SESSION_DURATION - time) / SESSION_DURATION, lob)
                    if quote is not None:
                        exchange.add_order(quote, False)
                        with _quiet():
                            trade = exchange.process_order2(time, quote, None, False)
                        if trade is not None:
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

            final_mid = float(np.mean([t['price'] for t in exchange.tape[-20:]])) \
                        if exchange.tape else 100.0
            for trader in traders.values():
                if trader.ttype == 'CUSTOM':
                    profits.append(trader.cash + trader.inventory * final_mid - INIT_CASH)

    return float(np.mean(profits)) if profits else -10000.0


@use_named_args(_search_space)
def _bo_objective(**params):
    mean_profit = _bo_objective_simulation(
        params['alpha'], params['position_limit'], params['volatility_threshold'])

    _bo_log.append({
        'iteration':            len(_bo_log) + 1,
        'alpha':                round(params['alpha'], 6),
        'position_limit':       int(params['position_limit']),
        'volatility_threshold': round(params['volatility_threshold'], 6),
        'mean_profit':          round(mean_profit, 4),
        'timestamp':            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    })
    pd.DataFrame(_bo_log).to_csv(BAYES_LOG, index=False)

    elapsed = (datetime.now() - _bo_start).total_seconds()
    print('  [%03d/100] alpha=%.4f pos=%3d vol=%.3f -> profit=%+.1f  (%.1f min)' % (
        len(_bo_log), params['alpha'], int(params['position_limit']),
        params['volatility_threshold'], mean_profit, elapsed / 60))

    return -mean_profit


def run_bayesian_optimisation():
    global CUSTOM_PARAMS, _bo_start
    _bo_log.clear()
    _bo_start = datetime.now()

    print('\nBayesian Optimisation')
    print('Search space: alpha in [0.001, 0.30], position_limit in [20, 800], '
          'volatility_threshold in [3.0, 15.0]')
    print('100 iterations, 20 random initialisations, Expected Improvement acquisition')
    print('Started: %s\n' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    result = gp_minimize(
        func=_bo_objective,
        dimensions=_search_space,
        n_calls=100,
        n_initial_points=20,
        acq_func='EI',
        random_state=42,
        noise=0.1,
        verbose=False,
    )

    opt_alpha   = float(result.x[0])
    opt_pos_lim = int(result.x[1])
    opt_vol     = float(result.x[2])
    opt_profit  = -result.fun

    print('\nOptimal parameters:')
    print('  alpha                = %.6f' % opt_alpha)
    print('  position_limit       = %d'   % opt_pos_lim)
    print('  volatility_threshold = %.6f' % opt_vol)
    print('  expected mean profit = %.2f ECU' % opt_profit)

    pd.DataFrame(_bo_log).to_csv(BAYES_LOG, index=False)
    pd.DataFrame([{
        'alpha': round(opt_alpha, 6),
        'position_limit': opt_pos_lim,
        'volatility_threshold': round(opt_vol, 6),
        'expected_mean_profit': round(opt_profit, 4),
        'n_calls': 100,
        'n_initial_points': 20,
        'acquisition_fn': 'EI',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }]).to_csv(BAYES_OPT, index=False)

    CUSTOM_PARAMS['alpha']                = opt_alpha
    CUSTOM_PARAMS['position_limit']       = opt_pos_lim
    CUSTOM_PARAMS['volatility_threshold'] = opt_vol

    print('\nOptimal parameters saved to data/optimal_parameters.csv')
    return opt_alpha, opt_pos_lim, opt_vol


def run_main_simulation():
    os.makedirs(DATA_DIR, exist_ok=True)

    results_fields    = ['trial', 'condition', 'tid', 'ttype',
                         'final_profit', 'n_trades', 'final_inventory']
    inv_fields        = ['trial', 'condition', 'timestep', 'ttype', 'mean_inv']
    checkpoint_fields = ['condition', 'trial', 'timestamp']

    done      = completed_trials(CHECKPOINT)
    total     = len(CONDITIONS) * N_TRIALS
    completed = len(done)

    print('\nMain Simulation')
    print('Custom params: alpha=%.6f, position_limit=%d, volatility_threshold=%.6f' % (
        CUSTOM_PARAMS['alpha'], CUSTOM_PARAMS['position_limit'],
        CUSTOM_PARAMS['volatility_threshold']))
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
            results, inv_snaps = run_single_trial(condition, trial, CUSTOM_PARAMS)
            append_rows(RESULTS_OUT,  results,   results_fields)
            append_rows(INV_OUT,      inv_snaps,  inv_fields)
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

    print('\nSimulation complete. Results written to data/')

    # Summary table
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
    print('BSE Simulation Pipeline')
    print('Started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    run_bayesian_optimisation()

    print('\nCustom Trader parameters for main run:')
    for k, v in CUSTOM_PARAMS.items():
        print('  %s = %s' % (k, v))

    run_main_simulation()

    print('\nDone.')
