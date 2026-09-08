# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:22:21 2026

@author: alesc
"""

import tensorflow as tf
import numpy as np
from scipy.stats import norm

#from src.config import num_steps, K, brokerage_fee, r_riskfree

r_riskfree = 0.045 
num_steps = 30
K = 1.0            
brokerage_fee = 0.001


def naive_pnl(test_data, n_steps=num_steps, K=K, fee=brokerage_fee, return_deltas=False):
    """delta_t = 1 if S_t > K else 0. No training, no memory beyond the last delta
    (needed only to price the transaction cost of flipping in/out of the position)."""
    S = tf.cast(test_data, tf.float32)
    n = S.shape[0]
    delta = tf.zeros((n, 1))
    pnl = tf.zeros((n, 1))
    deltas = []

    for t in range(n_steps):
        S_t = S[:, t:t+1]
        new_delta = tf.where(S_t > K, 1.0, 0.0)
        trade = new_delta - delta
        transaction_costs = tf.abs(trade) * S_t * fee
        price_change = S[:, t+1:t+2] - S_t
        pnl += (new_delta * price_change) - transaction_costs
        delta = new_delta
        deltas.append(new_delta.numpy().flatten())

    payoff = tf.maximum(S[:, -1:] - K, 0.0)
    pnl -= payoff

    if return_deltas:
        return pnl.numpy().flatten(), np.array(deltas).T
    return pnl.numpy().flatten()


def unhedged_pnl(test_data, K=K):
    """delta ≡ 0: sells the option and does nothing until maturity. The floor
    every hedging strategy below needs to clear."""
    S = tf.cast(test_data, tf.float32).numpy()
    return -np.maximum(S[:, -1] - K, 0.0)

def bs_delta(S, K, tau, r, sigma):
    tau = np.maximum(tau, 1e-6)   # avoid /0 at the very last step
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)


def bs_pnl(test_data, sigma, n_steps=num_steps, K=K, fee=brokerage_fee, r=r_riskfree,
           return_deltas=False):
    S = tf.cast(test_data, tf.float32).numpy()
    n = S.shape[0]
    S_seq = S[:, :n_steps]

    tau = np.array([(n_steps - t) / 252.0 for t in range(n_steps)])[np.newaxis, :]
    delta = bs_delta(S_seq, K, tau, r, sigma)
    delta_prev = np.hstack([np.zeros((n, 1)), delta[:, :-1]])

    price_change = S[:, 1:n_steps + 1] - S_seq
    transaction_costs = np.abs(delta - delta_prev) * S_seq * fee
    pnl = np.sum(delta * price_change - transaction_costs, axis=1) - np.maximum(S[:, -1] - K, 0.0)

    if return_deltas:
        return pnl, delta
    return pnl
