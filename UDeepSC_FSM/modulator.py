# -*- coding:utf-8 -*-
# @Time: 2023/5/28 15:49

import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

def qam_mod(M):
    """
        Generate M-QAM mapping table and demapping table.

        Parameters
        ----------
        M: int. Modulation order, must be a positive integer power of 2 and a perfect square number, or one of 8 and 32.
        Returns
        -------
        map_table: dict. M-QAM mapping table.
        demap_table: dict.M-QAM demapping table
    """
    sqrtM = int(math.sqrt(M))
    assert (sqrtM ** 2 == M and M & (M-1) == 0) or (M == 32) or (M == 8),\
        "M must be a positive integer power of 2 and a perfect square number, or one of 8 and 32."
    if M == 8:
        graycode = np.array([0, 1, 3, 7, 5, 4, 6, 2])
        constellation = [(-2-2j), (-2+0j), (-2+2j), (0+2j), (2+2j), (2+0j), (2-2j), (0-2j)]
    elif M == 32:
        temp1 = np.bitwise_xor(np.arange(8), np.right_shift(np.arange(8), 1))
        temp2 = np.bitwise_xor(np.arange(4), np.right_shift(np.arange(4), 1))
        graycode = np.zeros(M, dtype=int)
        num = 0
        for i in temp1:
            for j in temp2:
                graycode[num] = 4 * i + j
                num += 1
        constellation = [(-7 - 3j) + 2 * (x + y * 1j) for x, y in np.ndindex(8, 4)]
    else:
        temp = np.bitwise_xor(np.arange(sqrtM), np.right_shift(np.arange(sqrtM), 1))
        graycode = np.zeros(M, dtype=int)
        num = 0
        for i in temp:
            for j in temp:
                graycode[num] = sqrtM * i + j
                num += 1
        constellation = [-(sqrtM-1)*(1+1j) + 2*(x+y*1j) for x, y in np.ndindex(sqrtM, sqrtM)]
    map_table = dict(zip(graycode, constellation))
    demap_table = {v: k for k, v in map_table.items()}
    return map_table, demap_table

def qam_mapper(bits, map_table):
    """
        Map coded bits into symbols using M-QAM technique.

        Parameters
        ----------
        bits: array(num_bit, ). Coded bits to be modulated.
        map_table: dict. M-QAM mapping table.
        Returns
        -------
        syms: array(num_symbol, ). Modulated symbols to be transmitted.
    """
    M = len(map_table)
    bits = np.reshape(bits, (-1, ))
    nbits = int(math.log2(M))
    if len(bits) % nbits != 0:
        bits = np.pad(bits, (0, nbits - len(bits) % nbits), constant_values=(0, 0))
    bit_blocks = np.reshape(bits, (-1, nbits))  # divide bits into bit blocks
    blocks_bin = [''.join(str(_) for _ in block) for block in bit_blocks]
    blocks_dec = [int(block, 2) for block in blocks_bin]  # binary to decimal
    syms = np.array([map_table[block] for block in blocks_dec])
    return syms


def qam_demapper(syms, demap_table):
    """
        Demap received symbols into digital bits according to M-QAM mapping table.

        Parameters
        ----------
        syms: array(num_bits, ). Received symbols with channel noise.
        demap_table: dict. M-QAM demapping table.
        Returns
        -------
        bits: array(num_bit, ). Demodulated bits.
    """
    M = len(demap_table)
    nbits = int(math.log2(M))
    constellation = np.array([x for x in demap_table.keys()])
    dists = np.abs(syms.reshape(-1, 1) - constellation.reshape(1, -1))
    const_index = dists.argmin(axis=1)
    hardDecision = constellation[const_index]
    bit_blocks = [bin(demap_table[C])[2:].rjust(nbits, '0') for C in hardDecision]
    bits_str = ''.join(block for block in bit_blocks)
    bits = np.array([int(_) for _ in bits_str])
    return bits


def channel_Awgn(tx_signal, snr, output_power=False):
    """
        AWGN channel model.

        Parameters
        ----------
        tx_signal: array(num_symbols, ). Signal to be transmitted.
        snr: int. SNR at the receiver side.
        output_power: bool, default False. Whether to print signal power and noise power.
        Returns
        -------
        bits: array(num_bit, ). Demodulated bits.
    """
    signal_power = np.mean(abs(tx_signal ** 2))
    # print(signal_power)
    n_var = signal_power * 10 ** (- snr / 10)  # calculate noise power based on signal power and SNR
    if output_power:
        print(f"RX Signal power: {signal_power: .4f}. Noise power: {n_var: .4f}")
    # Generate complex noise
    noise = math.sqrt(n_var/2) * (np.random.randn(*tx_signal.shape)+1j*np.random.randn(*tx_signal.shape))
    return tx_signal + noise


def channel_Rayleigh(tx_signal, snr, output_power=False):
    """
        Rayleigh channel model.

        Parameters
        ----------
        tx_signal: array(num_symbols, ). Signal to be transmitted.
        snr: int. SNR at the receiver side.
        output_power: bool, default False. Whether to print signal power and noise power.
        Returns
        -------
        bits: array(num_bit, ). Demodulated bits.
    """
    shape = tx_signal.shape
    sigma = math.sqrt(1 / 2)
    H = np.random.normal(0.0, sigma, size=shape) + 1j * np.random.normal(0.0, sigma, size=shape)
    Tx_sig = tx_signal * H
    Rx_sig = channel_Awgn(Tx_sig, snr, output_power=output_power)
    # Channel estimation
    Rx_sig = Rx_sig / H
    return Rx_sig


def channel_Rician(tx_signal, snr, output_power=False, K=1):
    """
        Rician channel model.

        Parameters
        ----------
        tx_signal: array(num_symbols, ). Signal to be transmitted.
        snr: int. SNR at the receiver side.
        output_power: bool, default False. Whether to print signal power and noise power.
        Returns
        -------
        bits: array(num_bit, ). Demodulated bits.
    """
    shape = tx_signal.shape
    mean = math.sqrt(K / (K + 1))
    std = math.sqrt(1 / (K + 1))
    H = np.random.normal(mean, std, size=shape) + 1j * np.random.normal(mean, std, size=shape)
    Tx_sig = tx_signal * H
    Rx_sig = channel_Awgn(Tx_sig, snr, output_power=output_power)
    # Channel estimation
    Rx_sig = Rx_sig / H
    return Rx_sig


def bit_error_rate(tx_bits, rx_bits):
    return np.sum(abs(tx_bits - rx_bits)) / len(tx_bits)

def q_func(x):
    Qx = 0.5*math.erfc(x/math.sqrt(2))
    return Qx
