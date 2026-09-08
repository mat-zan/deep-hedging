# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:36:11 2026

@author: alesc
"""

import tensorflow as tf
import numpy as np
import keras
import random

#from src.config import num_steps, K, brokerage_fee, r_riskfree, tau_seq, Risk_Aversion, MAX_EPOCHS, PATIENCE 
from metrics import entropic_loss
from Train import train_mlp_cfg, train_rnn_cfg, train_gru_cfg, train_logistic_cfg, train_lstm_cfg
from Baselines import bs_pnl, naive_pnl, unhedged_pnl
from metrics import sweep_metrics


Risk_Aversion = 10
num_steps = 30
K = 1.0            
brokerage_fee = 0.001
tau_seq = tf.constant([(num_steps - t) / num_steps for t in range(num_steps)], dtype=tf.float32)
MAX_EPOCHS = 30
PATIENCE   = 5
sweep_epochs = 20


def eval_mlp_with_deltas(model, test_data, n_steps=num_steps, K=K, fee=brokerage_fee):
    S = tf.cast(test_data, tf.float32)
    n = S.shape[0]
    delta = tf.zeros((n, 1))
    pnl = tf.zeros((n, 1))
    deltas = []

    for t in range(n_steps):
        S_t = S[:, t:t+1]
        tau_t = tf.fill((n, 1), (n_steps - t) / n_steps)
        new_delta = model(tf.concat([delta, S_t, tau_t], axis=1), training=False)
        pnl += new_delta * (S[:, t+1:t+2] - S_t) - tf.abs(new_delta - delta) * S_t * fee
        delta = new_delta
        deltas.append(new_delta.numpy().flatten())

    pnl -= tf.maximum(S[:, -1:] - K, 0.0)
    return pnl.numpy().flatten(), np.array(deltas).T


def eval_rnn_with_deltas(model, test_data, n_steps=num_steps, K=K, fee=brokerage_fee):
    S = tf.cast(test_data, tf.float32)
    S_seq = S[:, :n_steps]
    # local tau grid, NOT the global 30-step tau_seq: this function must also work
    # when called with n_steps != 30 from the maturity sweep in Part 9.
    local_tau_seq = tf.constant([(n_steps - t) / n_steps for t in range(n_steps)], dtype=tf.float32)
    inputs = tf.stack([S_seq, tf.broadcast_to(local_tau_seq, tf.shape(S_seq))], axis=-1)

    delta_seq = model(inputs, training=False)[:, :, 0]
    delta_prev = tf.concat([tf.zeros_like(S_seq[:, 0:1]), delta_seq[:, :-1]], axis=1)

    price_change = S[:, 1:n_steps+1] - S_seq
    transaction_costs = tf.abs(delta_seq - delta_prev) * S_seq * fee
    pnl = tf.reduce_sum(delta_seq * price_change - transaction_costs, axis=1, keepdims=True)
    pnl -= tf.maximum(S[:, -1:] - K, 0.0)

    return pnl.numpy().flatten(), delta_seq.numpy()



def pnl_paths_mlp(model, S, fee=brokerage_fee, training=False):
    """Sequential: delta feeds back into the input. This feedback is the
    entire reason the MLP needs a Python-level time loop instead of a
    single vectorized forward pass — delta_t depends on the model's own
    delta_{t-1}, which a plain feedforward layer can't see any other way."""
    delta = tf.zeros_like(S[:, 0:1])
    pnl   = tf.zeros_like(S[:, 0:1])

    for t in range(num_steps):
        S_t   = S[:, t:t+1]
        tau_t = tf.ones_like(S_t) * ((num_steps - t) / num_steps)
        new_delta = model(tf.concat([delta, S_t, tau_t], axis=1), training=training)
        pnl += new_delta * (S[:, t+1:t+2] - S_t) - tf.abs(new_delta - delta) * S_t * fee
        delta = new_delta

    return pnl - tf.maximum(S[:, -1:] - K, 0.0)


def pnl_paths_rnn(model, S, fee=brokerage_fee, training=False):
    """One forward pass over the whole path, vectorised. Works for both
    GRU and LSTM since return_sequences=True gives us delta_t for every
    t in one call."""
    S_seq  = S[:, :num_steps]
    inputs = tf.stack([S_seq, tf.broadcast_to(tau_seq, tf.shape(S_seq))], axis=-1)

    delta_seq  = model(inputs, training=training)[:, :, 0]
    delta_prev = tf.concat([tf.zeros_like(S_seq[:, 0:1]), delta_seq[:, :-1]], axis=1)

    pnl = tf.reduce_sum(delta_seq * (S[:, 1:num_steps+1] - S_seq)
                        - tf.abs(delta_seq - delta_prev) * S_seq * fee,
                        axis=1, keepdims=True)

    return pnl - tf.maximum(S[:, -1:] - K, 0.0)




def logistic_pnl(hedger, test_data, n_steps=num_steps, K=K, fee=brokerage_fee, return_deltas=False):
    """The single evaluation function for the logistic hedger, reused by Part 7 (out-of-sample),
    Part 8 (day-by-day) and Part 9 (sweeps) -- one implementation instead of three, so the
    feature-order bug can't reappear in one call site while being correct in another."""
    S = tf.cast(test_data, tf.float32)
    n = S.shape[0]
    delta = tf.zeros((n, 1))
    pnl = tf.zeros((n, 1))
    deltas = []

    for t in range(n_steps):
        S_t = S[:, t:t+1]
        tau_t = tf.fill((n, 1), (n_steps - t) / n_steps)
        new_delta = hedger(tf.concat([delta, S_t, tau_t], axis=1), training=False)  # [delta_prev, S, tau]
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


def run_config(train_data, test_data,BEST, mlp_features, rnn_features, n_steps=num_steps, K=K, fee=brokerage_fee,
               risk_aversion=Risk_Aversion, sigma=None, epochs=sweep_epochs):
    """Retrains MLP, GRU, LSTM, Logistic on `train_data` at the given (n_steps, fee,
    risk_aversion), evaluates all 8 strategies on `test_data`, returns one row per model.
    sigma defaults to the historical volatility of `train_data` itself, so a maturity or
    fee sweep doesn't silently keep using Part 6's original 30-day sigma."""
    if sigma is None:
        train_lr = np.log(train_data[:, 1:] / train_data[:, :-1])
        sigma = float(np.std(train_lr) * np.sqrt(252))
    leland_sig = float(np.sqrt(sigma**2 * (1 + np.sqrt(2/np.pi) * (fee / (sigma * np.sqrt(1/252))))))

    mlp = train_mlp_cfg(train_data, n_steps, K, fee, risk_aversion, BEST, mlp_features, epochs=epochs)
    gru = train_gru_cfg(train_data, n_steps, K, fee, risk_aversion, BEST, rnn_features,  epochs=epochs)
    lstm = train_lstm_cfg(train_data, n_steps, K, fee, risk_aversion,BEST, rnn_features, epochs=epochs)
    log_model = train_logistic_cfg(train_data, n_steps, K, fee, risk_aversion, mlp_features, epochs=epochs)

    pnl_mlp_, d_mlp_   = eval_mlp_with_deltas(mlp, test_data, n_steps, K, fee)
    pnl_gru_, d_gru_   = eval_rnn_with_deltas(gru, test_data, n_steps, K, fee)
    pnl_lstm_, d_lstm_ = eval_rnn_with_deltas(lstm, test_data, n_steps, K, fee)
    pnl_log_, d_log_   = logistic_pnl(log_model, test_data, n_steps, K, fee, return_deltas=True)# Metti 'sigma' al suo posto oppure usa i nomi espliciti per tutto. Così è a prova di bomba:
    pnl_bs_, d_bs_     = bs_pnl(test_data, sigma=sigma, n_steps=n_steps, K=K, fee=fee, return_deltas=True)
    pnl_lel_, d_lel_   = bs_pnl(test_data, sigma=leland_sig, n_steps=n_steps, K=K, fee=fee, return_deltas= True)
    pnl_naive_, d_naive_ = naive_pnl(test_data, n_steps, K, fee, return_deltas=True)
    pnl_unh_ = unhedged_pnl(test_data, K)

    rows = []
    for name, pnl, deltas in [
        ("MLP", pnl_mlp_, d_mlp_), ("GRU", pnl_gru_, d_gru_), ("LSTM", pnl_lstm_, d_lstm_),
        ("Logistic", pnl_log_, d_log_), ("Black-Scholes", pnl_bs_, d_bs_),
        ("Leland", pnl_lel_, d_lel_), ("Naive", pnl_naive_, d_naive_),
        ("Unhedged", pnl_unh_, None),
    ]:
        row = sweep_metrics(pnl, risk_aversion, deltas)
        row["model"] = name
        rows.append(row)
    return rows



def bootstrap_entropic_ci(pnl_fn, data, n_boot=500, alpha=0.10, seed=42):
    """pnl_fn: callable(data) -> 1D pnl array. Resamples WINDOWS (rows), not
    individual return observations, so autocorrelation within a window is
    preserved -- only the cross-window sampling is randomized."""
    rng = np.random.default_rng(seed)
    n = len(data)
    losses = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        losses[b] = float(entropic_loss(pnl_fn(data[idx]), Risk_Aversion))
    lo, hi = np.percentile(losses, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(losses)), lo, hi