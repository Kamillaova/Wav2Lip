import librosa
import librosa.filters
import lws
import numpy as np
from scipy import signal

from hparams import hparams as hp


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav


def get_hop_size():
    hop_size = hp.hop_size
    if hop_size is None:
        assert hp.frame_shift_ms is not None
        hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
    return hop_size


def melspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db

    if hp.signal_normalization:
        return _normalize(S)
    return S


def _lws_processor():
    return lws.lws(hp.n_fft, get_hop_size(), fftsize=hp.win_size, mode="speech")


def _stft(y):
    if hp.use_lws:
        return _lws_processor().stft(y).T
    else:
        return librosa.stft(
            y=y, n_fft=hp.n_fft, hop_length=get_hop_size(), win_length=hp.win_size
        )


##########################################################
# Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
    """Compute number of time frames of spectrogram"""
    pad = fsize - fshift
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def pad_lr(x, fsize, fshift):
    """Compute left and right padding"""
    M = num_frames(len(x), fsize, fshift)
    pad = fsize - fshift
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


##########################################################


# Conversions
def _build_mel_basis():
    return librosa.filters.mel(
        sr=hp.sample_rate,
        n_fft=hp.n_fft,
        n_mels=hp.num_mels,
        fmin=hp.fmin,
        fmax=hp.fmax,
    )


_mel_basis = _build_mel_basis()


def _linear_to_mel(spectrogram):
    return np.dot(_mel_basis, spectrogram)


def _amp_to_db(amp):
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, amp))


def _db_to_amp(db):
    return np.power(10.0, db * 0.05)


def _normalize(level):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return np.clip(
                (2 * hp.max_abs_value)
                * ((level - hp.min_level_db) / (-hp.min_level_db))
                - hp.max_abs_value,
                -hp.max_abs_value,
                hp.max_abs_value,
            )
        else:
            return np.clip(
                hp.max_abs_value * ((level - hp.min_level_db) / (-hp.min_level_db)),
                0,
                hp.max_abs_value,
            )

    if hp.symmetric_mels:
        return (2 * hp.max_abs_value) * (
            (level - hp.min_level_db) / (-hp.min_level_db)
        ) - hp.max_abs_value
    else:
        return hp.max_abs_value * ((level - hp.min_level_db) / (-hp.min_level_db))


def _denormalize(D):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return (
                (np.clip(D, -hp.max_abs_value, hp.max_abs_value) + hp.max_abs_value)
                * -hp.min_level_db
                / (2 * hp.max_abs_value)
            ) + hp.min_level_db
        else:
            return (
                np.clip(D, 0, hp.max_abs_value) * -hp.min_level_db / hp.max_abs_value
            ) + hp.min_level_db

    if hp.symmetric_mels:
        return (
            (D + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value)
        ) + hp.min_level_db
    else:
        return (D * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db
