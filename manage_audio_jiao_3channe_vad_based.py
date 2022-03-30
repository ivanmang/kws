from __future__ import print_function
import argparse
import os
import random
import sys
import wave
import decimal

import librosa
import numpy as np
import pcen
import pyaudio
import signal
from scipy import signal


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def set_speech_format(f):
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(16000)


class AudioPreprocessor(object):
    def __init__(self, sr=8000, n_dct_filters=32, n_mels=32, f_max=4000, f_min=20, n_fft=256, hop_ms=30):
        super().__init__()
        self.n_mels = n_mels
        self.dct_filters = librosa.filters.dct(n_dct_filters, n_mels)
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min
        self.n_fft = n_fft
        self.hop_length = sr // 1000 * hop_ms
        self.pcen_transform = pcen.StreamingPCENTransform(n_mels=n_mels, n_fft=n_fft, hop_length=self.hop_length,
                                                          trainable=True)
        self.d_max = 43.79841613769531
        self.d_min = -508.64208984375

        self.bits = 16
        self.point_bits = 0
        self.value = np.power(2, self.point_bits)
        self.d_min_value = 0.5 ** self.point_bits
        self.inv_scale = 1

        self.log_d_max = 5
        self.log_bits = 16
        self.log_point_bits = 0
        self.log_bits_value = np.power(2, self.log_point_bits)
        # self.log_scale = 1
        self.inv_log_scale = round(255 / self.log_d_max * self.log_bits_value) / self.log_bits_value

    def quantize_tensor(self, x, num_bits=16):
        qmin = 0.
        qmax = 2. ** num_bits - 1.
        min_val, max_val = x.min(), x.max()
        scale = (max_val - min_val) / (qmax - qmin)

        initial_zero_point = qmin - min_val / scale

        zero_point = 0
        if initial_zero_point < qmin:
            zero_point = qmin
        elif initial_zero_point > qmax:
            zero_point = qmax
        else:
            zero_point = initial_zero_point

        zero_point = int(zero_point)
        q_x = zero_point + x / scale
        q_x = np.round(np.clip(q_x, qmin, qmax))
        return q_x

    def compute_mfccs(self, data):

        numframes = 32
        n_fft = 256
        n_field = 250
        frames = np.zeros([numframes, n_fft])
        int_data = np.int16(data * 32768)
        converted = int_data
        for i in range(int(numframes)):
            for j in range(int(n_field)):
                frames[i, j] = converted[i * n_field + j]

        frame_window = frames.T
        # print(frame_window.shape)
        # print(converted)
        # FFT#####################################################################################
        tot_layer = 8
        n_fft = n_fft
        nfft = n_fft
        Wn = np.zeros([nfft // 2], dtype=complex)
        Wn_around = np.zeros([nfft // 2], dtype=complex)
        Wn_r_around = np.zeros([nfft // 2])
        Wn_i_around = np.zeros([nfft // 2])
        Wn_r_mem = np.zeros([nfft // 2])
        Wn_i_mem = np.zeros([nfft // 2])

        for n in range(0, nfft // 2):
            Wn[n] = np.exp(-2 * np.pi * 1j * n / nfft)
            Wn_r_round = round(Wn[n].real * 32768)
            Wn_i_round = round(Wn[n].imag * 32768)
            if Wn_r_round >= 32768:
                Wn_r_around[n] = 32768 - 1
            elif Wn_r_round <= -32768:
                Wn_r_around[n] = -32768
            else:
                Wn_r_around[n] = Wn_r_round
            if Wn_i_round >= 32768:
                Wn_i_around[n] = 32768 - 1
            elif Wn_i_round <= -32768:
                Wn_i_around[n] = -32768
            else:
                Wn_i_around[n] = Wn_i_round

            Wn_around[n] = Wn_r_around[n] + 1j * Wn_i_around[n]

            Wn_r_mem[n] = np.uint64(Wn_r_around[n] + 65536) % 65536
            Wn_i_mem[n] = np.uint64(Wn_i_around[n] + 65536) % 65536
            Wn_mem = format(int(Wn_r_mem[n] * 65536 + Wn_i_mem[n]), '#010X')
            # print(Wn_mem[2:10])

        addr_fout = np.zeros([nfft])
        for k in range(0, nfft):
            k_tem = k
            kf = 0
            for kp in range(0, tot_layer):
                if (k_tem >= 2 ** (tot_layer - kp - 1)):
                    kf = kf + 2 ** kp
                    k_tem = k_tem - 2 ** (tot_layer - kp - 1)
                else:
                    kf = kf
                    k_tem = k_tem

            addr_fout[kf] = k

        tot_layer = 8
        addr_dr1 = np.zeros([tot_layer, nfft // 2])
        addr_dr2 = np.zeros([tot_layer, nfft // 2])
        addr_dwa = np.zeros([tot_layer, nfft // 2])
        addr_dwb = np.zeros([tot_layer, nfft // 2])
        addr_wr = np.zeros([tot_layer, nfft // 2])
        for layer in range(0, tot_layer):
            for k in range(0, nfft // 2):
                addr_dr1[layer, k] = (k // (nfft // (2 ** (layer + 1)))) * (2 ** (tot_layer - layer)) + (
                        k % (nfft // (2 ** (layer + 1))))
                addr_dr2[layer, k] = (k // (nfft // (2 ** (layer + 1)))) * (2 ** (tot_layer - layer)) + (
                        k % (nfft // (2 ** (layer + 1)))) + nfft // (2 ** (layer + 1))
                addr_dwa[layer, k] = (k // (nfft // (2 ** (layer + 1)))) * (2 ** (tot_layer - layer)) + (
                        k % (nfft // (2 ** (layer + 1))))
                addr_dwb[layer, k] = (k // (nfft // (2 ** (layer + 1)))) * (2 ** (tot_layer - layer)) + (
                        k % (nfft // (2 ** (layer + 1)))) + nfft // (2 ** (layer + 1))
                addr_wr[layer, k] = ((k * (2 ** layer)) % (2 ** (tot_layer - 1)))

        frame_window_T = frame_window.T
        numframes = 32
        numframes = numframes
        window_length_samples = 256
        bf_in = np.zeros([tot_layer, nfft], dtype=complex)
        bf_out = np.zeros([nfft], dtype=complex)
        fft_out = np.zeros([numframes, nfft], dtype=complex)
        power_fft = np.zeros([numframes, nfft])
        for i in range(0, numframes):
            addr_dr1 = np.zeros([tot_layer, nfft // 2])
            addr_dr2 = np.zeros([tot_layer, nfft // 2])
            addr_dw = np.zeros([tot_layer, nfft])
            addr_wr = np.zeros([tot_layer, nfft // 2])
            fft_frame = frame_window_T[i, 0:int(window_length_samples)]
            fft_frame_delay = np.pad(fft_frame, ((1), (0)))
            # print('fft_frame len:',fft_frame_delay.shape)
            fft_frame = fft_frame - (fft_frame_delay[0:nfft] - np.floor(fft_frame_delay[0:nfft] / 32))
            fft_frame = np.clip(fft_frame, -32768, 32767)
            # print(fft_frame)

            # for layer in range(0, tot_layer):
            for layer in range(0, 8):
                if layer == 0:
                    bf_in[layer, 0:int(window_length_samples)] = fft_frame
                else:
                    bf_in[layer, 0:nfft] = bf_out[0:nfft]

                for k in range(0, nfft // 2):
                    addr_dr1[layer, k] = (k // (nfft // (2 ** (layer + 1)))) * (2 ** (tot_layer - layer)) + (
                            k % (nfft // (2 ** (layer + 1))))
                    addr_dr2[layer, k] = (k // (nfft // (2 ** (layer + 1)))) * (2 ** (tot_layer - layer)) + (
                            k % (nfft // (2 ** (layer + 1)))) + nfft // (2 ** (layer + 1))
                    addr_dwa = (k // (nfft // (2 ** (layer + 1)))) * (2 ** (tot_layer - layer)) + (
                            k % (nfft // (2 ** (layer + 1))))
                    addr_dwb = (k // (nfft // (2 ** (layer + 1)))) * (2 ** (tot_layer - layer)) + (
                            k % (nfft // (2 ** (layer + 1)))) + nfft // (2 ** (layer + 1))
                    addr_wr[layer, k] = ((k * (2 ** layer)) % (2 ** (tot_layer - 1)))
                    if layer % 2 == 0:
                        bf_in_a = bf_in[layer, int(addr_dr1[layer, k])]
                        bf_in_b = bf_in[layer, int(addr_dr2[layer, k])]
                        add_apb_br = (bf_in_a + bf_in_b) / 1
                        add_amb_br = (bf_in_a - bf_in_b) / 1
                        add_apb_real = np.floor(add_apb_br.real)
                        add_apb_imag = np.floor(add_apb_br.imag)
                        add_amb_real = np.floor(add_amb_br.real)
                        add_amb_imag = np.floor(add_amb_br.imag)
                        if add_apb_real > 32767:
                            add_apb_real = 32767
                        elif add_apb_real < -32768:
                            add_apb_real = -32768
                        else:
                            add_apb_real = add_apb_real

                        if add_apb_imag > 32767:
                            add_apb_imag = 32767
                        elif add_apb_imag < -32768:
                            add_apb_imag = -32768
                        else:
                            add_apb_imag = add_apb_imag

                        if add_amb_real > 32767:
                            add_amb_real = 32767
                        elif add_amb_real < -32768:
                            add_amb_real = -32768
                        else:
                            add_amb_real = add_amb_real

                        if add_amb_imag > 32767:
                            add_amb_imag = 32767
                        elif add_amb_imag < -32768:
                            add_amb_imag = -32768
                        else:
                            add_amb_imag = add_amb_imag

                        add_apb = add_apb_real + 1j * add_apb_imag
                        add_amb = add_amb_real + 1j * add_amb_imag
                        weight = Wn_around[int(addr_wr[layer, k])]
                        add_amb_x_weight_rr = add_amb_real * weight.real
                        add_amb_x_weight_ri = add_amb_real * weight.imag
                        add_amb_x_weight_ir = add_amb_imag * weight.real
                        add_amb_x_weight_ii = add_amb_imag * weight.imag
                        tem_real = ((add_amb_x_weight_rr - add_amb_x_weight_ii) / 32768)
                        tem_imag = ((add_amb_x_weight_ri + add_amb_x_weight_ir) / 32768)
                        tem_real_round = np.floor(tem_real)
                        tem_imag_round = np.floor(tem_imag)
                        bf_out[int(addr_dwa)] = (add_apb_real + 1j * add_apb_imag)
                        bf_out[int(addr_dwb)] = tem_real_round + 1j * tem_imag_round
                    else:
                        bf_in_a = bf_in[layer, int(addr_dr1[layer, k])]
                        bf_in_b = bf_in[layer, int(addr_dr2[layer, k])]
                        add_apb_br = (bf_in_a + bf_in_b) / 2
                        add_amb_br = (bf_in_a - bf_in_b) / 2
                        add_apb_real = np.floor(add_apb_br.real)
                        add_apb_imag = np.floor(add_apb_br.imag)
                        add_amb_real = np.floor(add_amb_br.real)
                        add_amb_imag = np.floor(add_amb_br.imag)
                        if add_apb_real > 32767:
                            add_apb_real = 32767
                        elif add_apb_real < -32768:
                            add_apb_real = -32768
                        else:
                            add_apb_real = add_apb_real

                        if add_apb_imag > 32767:
                            add_apb_imag = 32767
                        elif add_apb_imag < -32768:
                            add_apb_imag = -32768
                        else:
                            add_apb_imag = add_apb_imag

                        if add_amb_real > 32767:
                            add_amb_real = 32767
                        elif add_amb_real < -32768:
                            add_amb_real = -32768
                        else:
                            add_amb_real = add_amb_real

                        if add_amb_imag > 32767:
                            add_amb_imag = 32767
                        elif add_amb_imag < -32768:
                            add_amb_imag = -32768
                        else:
                            add_amb_imag = add_amb_imag

                        add_apb = add_apb_real + 1j * add_apb_imag
                        add_amb = add_amb_real + 1j * add_amb_imag
                        weight = Wn_around[int(addr_wr[layer, k])]
                        add_amb_x_weight_rr = add_amb_real * weight.real
                        add_amb_x_weight_ri = add_amb_real * weight.imag
                        add_amb_x_weight_ir = add_amb_imag * weight.real
                        add_amb_x_weight_ii = add_amb_imag * weight.imag
                        tem_real = ((add_amb_x_weight_rr - add_amb_x_weight_ii) / 32768)
                        tem_imag = ((add_amb_x_weight_ri + add_amb_x_weight_ir) / 32768)
                        tem_real_round = np.floor(tem_real)
                        tem_imag_round = np.floor(tem_imag)
                        bf_out[int(addr_dwa)] = (add_apb_real + 1j * add_apb_imag)
                        bf_out[int(addr_dwb)] = tem_real_round + 1j * tem_imag_round

            for k in range(0, nfft):
                fft_out[i, int(addr_fout[k])] = bf_out[k] * 16

                power_fft[i, int(addr_fout[k])] = 256 * (
                        bf_out[k].real * bf_out[k].real + bf_out[k].imag * bf_out[k].imag)

        S_2 = power_fft[:, 0:int(1 + n_fft // 2)] // 32768  # 64 times
        ######################################################################################
        Sa = S_2.T / 1
        #  mel_basis = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.f_min, fmax=self.f_max)
        sr = 8000
        n_mels = 32
        f_max = 4000
        f_min = 20
        norm = 'slaney'
        htk = False

        weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float64)

        # Center freqs of each FFT bin
        fftfreqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        mel_f = librosa.mel_frequencies(n_mels + 2, fmin=f_min, fmax=f_max, htk=htk)

        fdiff = np.diff(mel_f)
        ramps = np.subtract.outer(mel_f, fftfreqs)

        for i in range(n_mels):
            # lower and upper slopes for all bins
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]

            # .. then intersect them with each other and zero
            weights[i] = np.maximum(0, np.minimum(lower, upper))

        if norm in (1, 'slaney'):
            # Slaney-style mel is scaled to be approx constant energy per channel
            enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
            weights *= enorm[:, np.newaxis]

        weights = np.real(weights)
        weights = weights.astype(np.float64)
        weights_stand = np.array(weights)
        for i in range(n_mels):
            # print('sum weights;',np.sum(weights[i, :]))
            weights_stand[i, :] = np.round(weights[i, :] * (32768 / np.sum(weights[i, :])))

        data_2a = (np.dot(weights_stand, Sa) // 32768)

        mfsc_bit = 16
        inter_num = 32
        log_num = 64
        dec_bit = 8
        log_int_quant = 2048 * 2
        dec_enlarge = round(log_int_quant * np.log(2 ** dec_bit))
        xi = np.arange(1, 2 + 1 / log_num, 1 / log_num)
        yi = np.round(np.log2(xi) * np.log(2) * log_int_quant)
        # print(data.shape)
        data = data_2a
        log_out = np.array(data)
        for fi in range(0, n_mels):
            for ti in range(0, numframes):
                X = max([data[fi, ti], 1])
                xa = 2 ** np.floor(np.log2(X))
                floor_inx = int(np.floor((X / xa - 1) * log_num))
                ceil_inx = floor_inx + 1
                dec_part_f = ((X / xa - 1) * log_num * inter_num - floor_inx * inter_num)
                dec_part = round(
                    decimal.Decimal(dec_part_f).quantize(decimal.Decimal(1), rounding=decimal.ROUND_HALF_UP))
                second_part = ((inter_num - dec_part) * yi[floor_inx] + (dec_part) * yi[ceil_inx]) / inter_num
                second_part_round = round(
                    decimal.Decimal(second_part).quantize(decimal.Decimal(1), rounding=decimal.ROUND_HALF_UP))
                log_result = (np.round(np.log2(xa) * np.log(2) * log_int_quant) + second_part_round)
                log_out[fi, ti] = log_result  # -dec_enlarge
        # data = log_out
        # print(log_out)
        feat = log_out
        NUMFRAMES = feat.shape[1]
        N = 5
        denominator = 2 * sum([i ** 2 for i in range(1, N + 1)])
        delta_feat = np.empty_like(feat)
        padded = np.pad(feat, ((0, 0), (N, N)), mode='edge')  # padded version of feat
        for t in range(NUMFRAMES):
            delta_feat[:, t] = np.floor(
                np.dot(padded[:, t: t + 2 * N + 1], np.arange(-N, N + 1)) * 149 / 2 ** 12) + 2 ** 15

        delta_delta_feat = np.empty_like(feat)
        delta_padded = np.pad(delta_feat, ((0, 0), (N, N)), mode='edge')  # padded version of feat
        for t in range(NUMFRAMES):
            delta_delta_feat[:, t] = np.floor(
                np.dot(delta_padded[:, t: t + 2 * N + 1], np.arange(-N, N + 1)) * 149 / 2 ** 12) + 2 ** 15

        mfsc_data = np.zeros([3, n_mels, NUMFRAMES])
        for fi in range(0, n_mels):
            for ti in range(0, numframes):
                mfsc_data[0, fi, ti] = round_half_up(feat[fi, ti] / 256)
                mfsc_data[1, fi, ti] = round_half_up(delta_feat[fi, ti] / 256)
                mfsc_data[2, fi, ti] = round_half_up(delta_delta_feat[fi, ti] / 256)

        return mfsc_data


class AudioSnippet(object):
    _dct_filters = librosa.filters.dct(40, 40)

    def __init__(self, byte_data=b"", dtype=np.int16):
        self.byte_data = byte_data
        self.dtype = dtype
        self._compute_amps()

    def save(self, filename):
        with wave.open(filename, "wb") as f:
            set_speech_format(f)
            f.writeframes(self.byte_data)

    def generate_contrastive(self):
        snippet = self.copy()
        phoneme_chunks = snippet.chunk_phonemes()
        phoneme_chunks2 = snippet.chunk_phonemes(factor=0.8, group_threshold=500)
        joined_chunks = []
        for i in range(len(phoneme_chunks) - 1):
            joined_chunks.append(AudioSnippet.join([phoneme_chunks[i], phoneme_chunks[i + 1]]))
        if len(joined_chunks) == 1:
            joined_chunks = []
        if len(phoneme_chunks) == 1:
            phoneme_chunks = []
        if len(phoneme_chunks2) == 1:
            phoneme_chunks2 = []
        chunks = [c.copy() for c in phoneme_chunks2]
        for chunk_list in (phoneme_chunks, joined_chunks, phoneme_chunks2):
            for chunk in chunk_list:
                chunk.rand_pad(32000)
        for chunk in chunks:
            chunk.repeat_fill(32000)
            chunk.rand_pad(32000)
        chunks.extend(phoneme_chunks)
        chunks.extend(phoneme_chunks2)
        chunks.extend(joined_chunks)
        return chunks

    def chunk_phonemes(self, factor=1.0, group_threshold=1000):
        audio_data, _ = librosa.effects.trim(self.amplitudes, top_db=16)
        data = librosa.feature.melspectrogram(audio_data, sr=8000, n_mels=40, hop_length=160, n_fft=256, fmin=20,
                                              fmax=4000)
        data[data > 0] = np.log(data[data > 0])
        # data = [np.matmul(AudioSnippet._dct_filters, x) for x in np.split(data, data.shape[1], axis=1)]
        data = np.array(data, order="F").squeeze(2).astype(np.float32)
        data = data[:, 1:25]
        a = []
        for i in range(data.shape[0] - 1):
            a.append(np.linalg.norm(data[i] - data[i + 1]))
        a = np.array(a)
        q75, q25 = np.percentile(a, [75, 25])
        segments = 160 * np.arange(a.shape[0])[a > q75 + factor * (q75 - q25)]
        segments = np.append(segments, [len(audio_data)])
        delete_idx = []
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                if segments[j] - segments[i] < group_threshold:
                    delete_idx.append(j)
                else:
                    i = j - 1
                    break
        segments = np.delete(segments, delete_idx)
        audio_segments = [audio_data[segments[i]:segments[i + 1]] for i in range(len(segments) - 1)]
        audio_segments = [AudioSnippet.from_amps(seg) for seg in audio_segments]
        return audio_segments

    @staticmethod
    def join(snippets):
        snippet = AudioSnippet(dtype=snippets[0].dtype)
        for s in snippets:
            snippet.append(s)
        return snippet

    def copy(self):
        return AudioSnippet(self.byte_data)

    def chunk(self, size, stride=1000):
        chunks = []
        i = 0
        while i + size < len(self.byte_data):
            chunks.append(AudioSnippet(self.byte_data[i:i + size]))
            i += stride
        return chunks

    def rand_pad(self, total_length, noise_level=0.001):
        space = total_length - len(self.byte_data)
        len_a = (random.randint(0, space)) // 2 * 2
        len_b = space - len_a
        self.byte_data = b"".join([b"".join([b"\x00"] * len_a), self.byte_data, b"".join([b"\x00"] * len_b)])
        self._compute_amps()

    def repeat_fill(self, length):
        n_times = max(1, length // len(self.byte_data))
        self.byte_data = b"".join([self.byte_data] * n_times)[:length]

    def trim_window(self, window_size):
        nbytes = len(self.byte_data) // len(self.amplitudes)
        window_size //= nbytes
        cc_window = np.ones(window_size)
        clip_energy = np.correlate(np.abs(self.amplitudes), cc_window)
        smooth_window_size = 1000
        smooth_window = np.ones(smooth_window_size)
        scale = len(self.amplitudes) / (len(self.amplitudes) - smooth_window_size + 1)
        clip_energy2 = np.correlate(clip_energy, smooth_window)
        window_i = int(np.argmax(clip_energy2) * scale)
        window_i = max(0, window_i - window_i % nbytes)
        self.amplitudes = self.amplitudes[window_i:window_i + window_size]
        window_i *= nbytes
        self.byte_data = self.byte_data[window_i:window_i + window_size * nbytes]

    def ltrim(self, limit=0.1):
        if not self.byte_data:
            return
        i = 0
        for i in range(len(self.amplitudes)):
            if self.amplitudes[i] > limit:
                break
        nbytes = len(self.byte_data) // len(self.amplitudes)
        i = max(0, i - i % nbytes)
        self.amplitudes = self.amplitudes[i:]
        self.byte_data = self.byte_data[i * nbytes:]
        return self

    def trim(self, limit=0.1):
        self.ltrim(limit)
        self.rtrim(limit)
        return self

    def rtrim(self, limit=0.05):
        if not self.byte_data:
            return
        i = len(self.amplitudes)
        for i in range(len(self.amplitudes) - 1, -1, -1):
            if self.amplitudes[i] > limit:
                break
        nbytes = len(self.byte_data) // len(self.amplitudes)
        i = min(len(self.amplitudes), i + (nbytes - i % nbytes))
        self.amplitudes = self.amplitudes[:i]
        self.byte_data = self.byte_data[:i * nbytes]
        return self

    @classmethod
    def from_amps(cls, amps, dtype=np.int16):
        byte_data = (np.iinfo(dtype).max * amps).astype(dtype).tobytes()
        return cls(byte_data)

    def _compute_amps(self):
        self.amplitudes = np.frombuffer(self.byte_data, self.dtype).astype(float) / np.iinfo(self.dtype).max

    def append(self, snippet):
        self.byte_data = b''.join([self.byte_data, snippet.byte_data])
        self._compute_amps()
        return self

    def amplitude_rms(self, start=0, end=-1):
        return np.sqrt(np.mean([a * a for a in self.amplitudes[start:end]]))


class AudioSnippetGenerator(object):
    def __init__(self, sr=16000, fmt=pyaudio.paInt16, chunk_size=1024, channels=1):
        self.sr = sr
        self.fmt = fmt
        self.channels = channels
        self.chunk_size = chunk_size
        self.stream = None

    def __enter__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.fmt, channels=self.channels, rate=self.sr, input=True,
                                      frames_per_buffer=self.chunk_size)
        return self

    def __exit__(self, *args):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        self.stream = None

    def __iter__(self):
        if not self.stream:
            raise ValueError("Audio stream isn't open")
        return self

    def __next__(self):
        return AudioSnippet(self.stream.read(self.chunk_size))


def print_sound_level():
    with AudioSnippetGenerator() as generator:
        for audio in generator:
            print("Sound level: {}".format(audio.amplitude_rms()), end="\r")


def generate_dir(directory):
    for filename in os.listdir(directory):
        fullpath = os.path.join(os.path.abspath(directory), filename)
        try:
            with wave.open(fullpath) as f:
                n_channels = f.getnchannels()
                width = f.getsampwidth()
                rate = f.getframerate()
                snippet = AudioSnippet(f.readframes(16000))
            for i, e in enumerate(snippet.generate_contrastive()):
                gen_name = os.path.join(directory, "gen-{}-{}".format(i, filename))
                e.save(gen_name)
            print("Generated from {}".format(filename))
        except (wave.Error, IsADirectoryError, PermissionError) as e:
            pass


def clean_dir(directory=".", cutoff_ms=1000):
    """Trims all audio in directory to the loudest window of length cutoff_ms. 1 second is consistent
    with the speech command dataset.

    Args:
        directory: The directory containing all the .wav files. Should have nothing but .wav
        cutoff_ms: The length of time to trim audio to in milliseconds
    """
    for filename in os.listdir(directory):
        fullpath = os.path.join(directory, filename)
        try:
            with wave.open(fullpath) as f:
                n_channels = f.getnchannels()
                width = f.getsampwidth()
                rate = f.getframerate()
                n_samples = int((cutoff_ms / 1000) * rate)
                snippet = AudioSnippet(f.readframes(10 * n_samples))
            snippet.trim_window(n_samples * width)
            with wave.open(fullpath, "w") as f:
                f.setnchannels(n_channels)
                f.setsampwidth(width)
                f.setframerate(rate)
                f.writeframes(snippet.byte_data)
            print("Trimmed {} to {} ms".format(filename, cutoff_ms))
        except (wave.Error, IsADirectoryError, PermissionError) as e:
            pass


def main():
    parser = argparse.ArgumentParser()
    commands = dict(trim=clean_dir, listen=print_sound_level)
    commands["generate-contrastive"] = generate_dir
    parser.add_argument("subcommand")

    def print_sub_commands():
        print("Subcommands: {}".format(", ".join(commands.keys())))

    if len(sys.argv) <= 1:
        print_sub_commands()
        return
    subcommand = sys.argv[1]
    if subcommand == "generate-contrastive":
        parser.add_argument(
            "directory",
            type=str,
            default=".",
            help="Generate from the directory's audio files")
        flags, _ = parser.parse_known_args()
        generate_dir(flags.directory)
    elif subcommand == "trim":
        parser.add_argument(
            "directory",
            type=str,
            nargs="?",
            default=".",
            help="Trim the directory's audio files")
        flags, _ = parser.parse_known_args()
        clean_dir(flags.directory)
    elif subcommand == "listen":
        print_sound_level()
    else:
        print_sub_commands()


if __name__ == "__main__":
    main()