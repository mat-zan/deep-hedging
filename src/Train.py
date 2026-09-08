# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:47:46 2026

@author: alesc
"""

import tensorflow as tf
import numpy as np
import keras
import random

#from src.config import num_steps, K, brokerage_fee, r_riskfree, tau_seq, Risk_Aversion, MAX_EPOCHS, PATIENCE, sweep_epochs 
from metrics import entropic_loss

Risk_Aversion = 10
num_steps = 30
K = 1.0            
brokerage_fee = 0.001
tau_seq = tf.constant([(num_steps - t) / num_steps for t in range(num_steps)], dtype=tf.float32)
MAX_EPOCHS = 30
PATIENCE   = 5
sweep_epochs = 20


@tf.function
def train_step_mlp(Batch,hedger_mlp, optimizer_mlp, Risk_Aversion=Risk_Aversion):

    # initialize delta and pnl for all
    # the 512 paths in the Batch
    delta = tf.zeros((512, 1))
    pnl = tf.zeros((512, 1))

    # automatic differentiation
    with tf.GradientTape() as tape:

        # iteration over 30 trading days
        for t in range(num_steps):

            # 1) price at time t (S_t)
            S_t = Batch[:, t:t+1]

            # 2) market state (three inputs)
            tau_t = tf.fill((512, 1), (num_steps - t) / num_steps)

            # 3) new delta with MLP hedger
            #    how much to sell or buy
            new_delta = hedger_mlp(tf.concat([delta, S_t, tau_t], axis=1), training=True)
            trade = new_delta - delta

            # 4) costs, gains and pnl
            transaction_costs = tf.abs(trade) * S_t * brokerage_fee
            price_change = Batch[:, t+1:t+2] - S_t
            pnl += (new_delta * price_change) - transaction_costs

            # 5) update delta
            delta = new_delta

        # payoff subtraction at maturity
        payoff = tf.maximum(Batch[:, -1:] - K, 0.0)
        pnl -= payoff

        # entropic loss
        loss = entropic_loss(pnl, Risk_Aversion)

    # Backpropagation Through Time + optimization step
    gradients = tape.gradient(loss, hedger_mlp.trainable_variables)
    optimizer_mlp.apply_gradients(zip(gradients, hedger_mlp.trainable_variables))

    return loss, tf.reduce_mean(pnl)




@tf.function
def train_step_gru(Batch, optimizer_gru, hedger_gru, Risk_Aversion=Risk_Aversion):

    S_seq = Batch[:, :num_steps]
    inputs = tf.stack([S_seq, tf.broadcast_to(tau_seq, tf.shape(S_seq))], axis=-1)

    with tf.GradientTape() as tape:

        # one forward pass -> a delta for every one of the 30 days
        delta_seq = hedger_gru(inputs, training=True)[:, :, 0]
        delta_prev = tf.concat([tf.zeros_like(S_seq[:, 0:1]), delta_seq[:, :-1]], axis=1)

        price_change = Batch[:, 1:num_steps+1] - S_seq
        transaction_costs = tf.abs(delta_seq - delta_prev) * S_seq * brokerage_fee

        pnl = tf.reduce_sum(delta_seq * price_change - transaction_costs, axis=1, keepdims=True)

        # payoff subtraction at maturity
        payoff = tf.maximum(Batch[:, -1:] - K, 0.0)
        pnl -= payoff

        # entropic loss
        loss = entropic_loss(pnl, Risk_Aversion)

    gradients = tape.gradient(loss, hedger_gru.trainable_variables)
    optimizer_gru.apply_gradients(zip(gradients, hedger_gru.trainable_variables))

    return loss, tf.reduce_mean(pnl)


@tf.function
def train_step_lstm(Batch, hedger_lstm, optimizer_lstm, Risk_Aversion=Risk_Aversion):

    S_seq = Batch[:, :num_steps]
    inputs = tf.stack([S_seq, tf.broadcast_to(tau_seq, tf.shape(S_seq))], axis=-1)

    with tf.GradientTape() as tape:

        delta_seq = hedger_lstm(inputs, training=True)[:, :, 0]
        delta_prev = tf.concat([tf.zeros_like(S_seq[:, 0:1]), delta_seq[:, :-1]], axis=1)

        price_change = Batch[:, 1:num_steps+1] - S_seq
        transaction_costs = tf.abs(delta_seq - delta_prev) * S_seq * brokerage_fee

        pnl = tf.reduce_sum(delta_seq * price_change - transaction_costs, axis=1, keepdims=True)

        payoff = tf.maximum(Batch[:, -1:] - K, 0.0)
        pnl -= payoff

        loss = entropic_loss(pnl, Risk_Aversion)

    gradients = tape.gradient(loss, hedger_lstm.trainable_variables)
    optimizer_lstm.apply_gradients(zip(gradients, hedger_lstm.trainable_variables))

    return loss, tf.reduce_mean(pnl)



@tf.function
def train_step_logistic(Batch, hedger_logistic, optimizer_logistic, Risk_Aversion=Risk_Aversion):
    delta = tf.zeros((512, 1))
    pnl = tf.zeros((512, 1))

    with tf.GradientTape() as tape:
        for t in range(num_steps):
            S_t = Batch[:, t:t+1]
            tau_t = tf.fill((512, 1), (num_steps - t) / num_steps)

            # feature order: [delta_prev, S_t, tau_t] -- identical to hedger_mlp's training,
            # and to logistic_pnl() below.
            new_delta = hedger_logistic(tf.concat([delta, S_t, tau_t], axis=1), training=True)
            trade = new_delta - delta
            transaction_costs = tf.abs(trade) * S_t * brokerage_fee
            price_change = Batch[:, t+1:t+2] - S_t
            pnl += (new_delta * price_change) - transaction_costs
            delta = new_delta

        payoff = tf.maximum(Batch[:, -1:] - K, 0.0)
        pnl -= payoff
        loss = entropic_loss(pnl, Risk_Aversion)

    gradients = tape.gradient(loss, hedger_logistic.trainable_variables)
    optimizer_logistic.apply_gradients(zip(gradients, hedger_logistic.trainable_variables))
    return loss, tf.reduce_mean(pnl)



def fit_one(kind, width, lr, BUILDERS, dataset, val_tf, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    make, pnl_fn = BUILDERS[kind]

    # same seed for every configuration, so the comparison is fair
    random.seed(42); np.random.seed(42); tf.random.set_seed(42)

    model = make(width)
    opt   = keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def step(Batch):
        with tf.GradientTape() as tape:
            loss = entropic_loss(pnl_fn(model, Batch, training=True), Risk_Aversion)
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables),
                                model.trainable_variables))
        return loss

    best_val, best_epoch, since_best, curve = np.inf, 0, 0, []

    for epoch in range(1, max_epochs + 1):
        for batch in dataset:
            step(tf.cast(batch, tf.float32))

        # entropic loss on the FULL validation set, in one shot
        v = float(entropic_loss(pnl_fn(model, val_tf), Risk_Aversion))
        curve.append(v)

        if v < best_val - 1e-6:
            best_val, best_epoch, since_best = v, epoch, 0
        else:
            since_best += 1
            if since_best >= patience:
                break

    return {"model": kind, "width": width, "lr": lr,
            "epochs": best_epoch, "val_entropic": best_val, "curve": curve}


#function for faster training in the variation parameters section


def train_mlp_cfg(train_data, n_steps, K, fee, risk_aversion, BEST, mlp_features, epochs=sweep_epochs,
                   width=None, lr=None):
    width = width or BEST["MLP"]["width"]
    lr = lr or BEST["MLP"]["lr"]
    ds = tf.data.Dataset.from_tensor_slices(train_data).shuffle(len(train_data)).batch(512, drop_remainder=True)
    model = keras.Sequential([
        keras.Input(shape=(3,)), keras.layers.Lambda(mlp_features),
        keras.layers.Dense(width, activation="relu"),
        keras.layers.Dense(width, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    opt = keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def step(Batch):
        delta = tf.zeros((Batch.shape[0], 1))
        pnl = tf.zeros((Batch.shape[0], 1))
        with tf.GradientTape() as tape:
            for t in range(n_steps):
                S_t = Batch[:, t:t+1]
                tau_t = tf.fill((Batch.shape[0], 1), (n_steps - t) / n_steps)
                new_delta = model(tf.concat([delta, S_t, tau_t], axis=1), training=True)
                pnl += new_delta * (Batch[:, t+1:t+2] - S_t) - tf.abs(new_delta - delta) * S_t * fee
                delta = new_delta
            pnl -= tf.maximum(Batch[:, -1:] - K, 0.0)
            loss = entropic_loss(pnl, risk_aversion)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

    # Il ciclo delle epoche ora chiama la funzione ultra-veloce
    for epoch in range(epochs):
        for batch in ds:
            step(tf.cast(batch, tf.float32))
    return model

def train_rnn_cfg(train_data, n_steps, K, fee, risk_aversion, epochs, width, lr, cell_type, rnn_features):
    ds = tf.data.Dataset.from_tensor_slices(train_data).shuffle(len(train_data)).batch(512, drop_remainder=True)
    rnn_layer = keras.layers.GRU(width, return_sequences=True) if cell_type == "GRU" \
        else keras.layers.LSTM(width, return_sequences=True)
    model = keras.Sequential([
        keras.Input(shape=(n_steps, 2)), keras.layers.Lambda(rnn_features),
        rnn_layer, keras.layers.Dense(1, activation="sigmoid"),
    ])
    opt = keras.optimizers.Adam(learning_rate=lr)
    local_tau_seq = tf.constant([(n_steps - t) / n_steps for t in range(n_steps)], dtype=tf.float32)

    # -- INIZIO BLOCCO OTTIMIZZATO PER GPU --
    @tf.function
    def step(batch):
        Batch = tf.cast(batch, tf.float32)
        S_seq = Batch[:, :n_steps]
        inputs = tf.stack([S_seq, tf.broadcast_to(local_tau_seq, tf.shape(S_seq))], axis=-1)
        with tf.GradientTape() as tape:
            delta_seq = model(inputs, training=True)[:, :, 0]
            delta_prev = tf.concat([tf.zeros_like(S_seq[:, 0:1]), delta_seq[:, :-1]], axis=1)
            pnl = tf.reduce_sum(delta_seq * (Batch[:, 1:n_steps+1] - S_seq)
                                - tf.abs(delta_seq - delta_prev) * S_seq * fee, axis=1, keepdims=True)
            pnl -= tf.maximum(Batch[:, -1:] - K, 0.0)
            loss = entropic_loss(pnl, risk_aversion)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
    # -- FINE BLOCCO OTTIMIZZATO --

    # Il ciclo delle epoche ora esegue il grafo pre-compilato
    for epoch in range(epochs):
        for batch in ds:
            step(batch)
            
    return model


def train_gru_cfg(train_data, n_steps, K, fee, risk_aversion, BEST, rnn_features, epochs=sweep_epochs, width=None, lr=None):
    return train_rnn_cfg(train_data, n_steps, K, fee, risk_aversion, epochs,
                         width or BEST["GRU"]["width"], 
                         lr or BEST["GRU"]["lr"], 
                         "GRU", 
                         rnn_features)

def train_lstm_cfg(train_data, n_steps, K, fee, risk_aversion, BEST, rnn_features, epochs=sweep_epochs, width=None, lr=None):
    return train_rnn_cfg(train_data, n_steps, K, fee, risk_aversion, epochs,
                         width or BEST["LSTM"]["width"], 
                         lr or BEST["LSTM"]["lr"], 
                         "LSTM", 
                         rnn_features)


def train_logistic_cfg(train_data, n_steps, K, fee, risk_aversion, mlp_features, epochs=sweep_epochs, lr=1e-3):
    ds = tf.data.Dataset.from_tensor_slices(train_data).shuffle(len(train_data)).batch(512, drop_remainder=True)
    model = keras.Sequential([keras.Input(shape=(3,)), keras.layers.Lambda(mlp_features),
                               keras.layers.Dense(1, activation="sigmoid")])
    opt = keras.optimizers.Adam(learning_rate=lr)

    for epoch in range(epochs):
        for batch in ds:
            Batch = tf.cast(batch, tf.float32)
            delta = tf.zeros((Batch.shape[0], 1)); pnl = tf.zeros((Batch.shape[0], 1))
            with tf.GradientTape() as tape:
                for t in range(n_steps):
                    S_t = Batch[:, t:t+1]
                    tau_t = tf.fill((Batch.shape[0], 1), (n_steps - t) / n_steps)
                    new_delta = model(tf.concat([delta, S_t, tau_t], axis=1), training=True)  # [delta, S, tau]
                    pnl += new_delta * (Batch[:, t+1:t+2] - S_t) - tf.abs(new_delta - delta) * S_t * fee
                    delta = new_delta
                pnl -= tf.maximum(Batch[:, -1:] - K, 0.0)
                loss = entropic_loss(pnl, risk_aversion)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
    return model
