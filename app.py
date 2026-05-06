from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
from uuid import uuid4
import hashlib
import time

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

key_store = {}

def generate_fingerprint(key_data):
    """Create a short fingerprint from key data."""
    if isinstance(key_data, bytes):
        data = key_data[:8]
    elif isinstance(key_data, np.ndarray):
        data = key_data.tobytes()[:8]
    elif isinstance(key_data, (int, float)):
        data = str(key_data).encode()[:8]
    else:
        data = str(key_data).encode()[:8]
    return data.hex().upper()

def store_key(method, key_data):
    key_id = uuid4().hex
    fingerprint = generate_fingerprint(key_data)
    key_store[key_id] = {
        'method': method,
        'key_data': key_data,
        'fingerprint': fingerprint
    }
    return key_id, fingerprint

def get_key(key_id):
    return key_store.get(key_id)

# ------------------------------------------------------------------
# Helper random generator
# ------------------------------------------------------------------
def get_rng(seed=None):
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, (int, float)):
        return np.random.default_rng(int(seed * 1e12))
    return np.random.default_rng(seed)

# ------------------------------------------------------------------
# SHAPE NORMALISATION
# ------------------------------------------------------------------
_ALIGN = 32

def _pad_to_aligned(img):
    h, w = img.shape
    ph = (_ALIGN - h % _ALIGN) % _ALIGN
    pw = (_ALIGN - w % _ALIGN) % _ALIGN
    if ph == 0 and pw == 0:
        return img, h, w
    padded = np.pad(img, ((0, ph), (0, pw)), mode='edge')
    return padded, h, w

# ------------------------------------------------------------------
# DWT 
# ------------------------------------------------------------------
def _haar_dwt_2d(img_f32):
    L = (img_f32[:, 0::2] + img_f32[:, 1::2]) * 0.5
    H = (img_f32[:, 0::2] - img_f32[:, 1::2]) * 0.5
    LL = (L[0::2, :] + L[1::2, :]) * 0.5
    LH = (L[0::2, :] - L[1::2, :]) * 0.5
    HL = (H[0::2, :] + H[1::2, :]) * 0.5
    HH = (H[0::2, :] - H[1::2, :]) * 0.5
    return LL, LH, HL, HH

def _haar_idwt_2d(LL, LH, HL, HH):
    h2, w2 = LL.shape
    L = np.empty((h2 * 2, w2), dtype=np.float32)
    L[0::2, :] = LL + LH
    L[1::2, :] = LL - LH
    H = np.empty_like(L)
    H[0::2, :] = HL + HH
    H[1::2, :] = HL - HH
    out = np.empty((h2 * 2, w2 * 2), dtype=np.float32)
    out[:, 0::2] = L + H
    out[:, 1::2] = L - H
    return out

def _chaos_perm(size, seed=0.5, r=3.99):
    rng = np.random.default_rng(int(seed * 1e15) ^ int(r * 1e10))
    return rng.permutation(size)

# ------------------------------------------------------------------
# DNA encryption (seed‑based, no decryption needed – encryption is symmetric)
# ------------------------------------------------------------------
def _dna_encrypt(img_flat, key_seed=None):
    data = img_flat.astype(np.uint8)
    bits = np.unpackbits(data)
    hi = bits[0::2]
    lo = bits[1::2]
    data_vals = (hi * 2 + lo).astype(np.uint8)
    
    if key_seed is None:
        rng = np.random.default_rng(42)
        key_vals = rng.integers(0, 4, size=len(data_vals), dtype=np.uint8)
    else:
        rng = np.random.default_rng(int(key_seed))
        key_vals = rng.integers(0, 4, size=len(data_vals), dtype=np.uint8)
    
    enc_vals = np.bitwise_xor(data_vals, key_vals)
    enc_hi = (enc_vals >> 1) & 1
    enc_lo = enc_vals & 1
    bits_out = np.empty(len(enc_vals) * 2, dtype=np.uint8)
    bits_out[0::2] = enc_hi
    bits_out[1::2] = enc_lo
    return np.packbits(bits_out)[:len(img_flat)]

# ------------------------------------------------------------------
# AES‑like (symmetric)
# ------------------------------------------------------------------
_SBOX = np.array([(i * 197 + 31) % 256 for i in range(256)], dtype=np.uint8)

def _aes_encrypt(img_flat, master_key=None):
    data = img_flat.astype(np.uint8).copy()
    n = len(data)
    if master_key is None:
        rng = np.random.default_rng(0xDEADBEEF)
        master_key = rng.integers(0, 256, size=256, dtype=np.uint8)
    for rnd in range(5):
        data = _SBOX[data]
        tail = n % 16
        main_n = n - tail
        if main_n > 0:
            block = data[:main_n].reshape(-1, 16)
            block = np.roll(block, -(rnd + 1), axis=1)
            data[:main_n] = block.ravel()
        rk = np.tile(np.roll(master_key, rnd * 37), (n // 256) + 1)[:n].astype(np.uint8)
        data = np.bitwise_xor(data, rk)
    return data

# ------------------------------------------------------------------
# Cellular Automata (non‑invertible – encryption only)
# ------------------------------------------------------------------
def _ca_mix(img_flat):
    bits = np.unpackbits(img_flat.astype(np.uint8))
    for _ in range(5):
        L = np.roll(bits, 1)
        R = np.roll(bits, -1)
        bits = np.bitwise_xor(L, np.bitwise_or(bits, R))
    return np.packbits(bits)[:len(img_flat)]

# ------------------------------------------------------------------
# Compressive Sensing (non‑invertible – encryption only)
# ------------------------------------------------------------------
def _cs_shuffle(img_flat):
    n = len(img_flat)
    signal = img_flat.astype(np.float32)
    rng = np.random.default_rng(42)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)
    flipped = signal * signs
    if n % 2:
        flipped = np.append(flipped, 0.0)
    measurements = flipped.reshape(-1, 2).sum(axis=1)
    repeated = np.repeat(measurements, 2)[:n]
    perm = _chaos_perm(n, seed=0.618, r=3.9997)
    scrambled = repeated[perm]
    mn, mx = scrambled.min(), scrambled.max()
    if mx - mn < 1e-6:
        return np.zeros(n, dtype=np.uint8)
    return ((scrambled - mn) / (mx - mn) * 255).astype(np.uint8)

# ------------------------------------------------------------------
# Rubik Cube (reversible)
# ------------------------------------------------------------------
def _rubik_permute(img_u8):
    h, w = img_u8.shape
    bs = _ALIGN
    out = img_u8.copy()
    src = img_u8
    for bi, i in enumerate(range(0, h, bs)):
        for bj, j in enumerate(range(0, w, bs)):
            ie, je = min(i + bs, h), min(j + bs, w)
            k = (bi + bj) % 4
            if k:
                out[i:ie, j:je] = np.rot90(src[i:ie, j:je], k=k)
    return out

# ------------------------------------------------------------------
# Core encryption dispatcher (no decryption)
# ------------------------------------------------------------------
def apply_encryption(img, method_name, key_data=None):
    orig_h, orig_w = img.shape
    padded, _, _ = _pad_to_aligned(img)
    h, w = padded.shape
    img_f = padded.astype(np.float32)
    flat = padded.flatten().astype(np.uint8)
    n = len(flat)

    if method_name == 'DCT':
        dct = cv2.dct(img_f)
        perm = _chaos_perm(dct.size, seed=key_data if key_data is not None else 0.5)
        scr = dct.ravel()[perm].reshape(dct.shape)
        result = cv2.idct(scr)

    elif method_name == 'DWT':
        LL, LH, HL, HH = _haar_dwt_2d(img_f)
        seed = key_data if key_data is not None else 0.6
        perm = _chaos_perm(LL.size, seed=seed)
        def _scr(b):
            return b.ravel()[perm].reshape(b.shape)
        result = _haar_idwt_2d(_scr(LL), _scr(LH), _scr(HL), _scr(HH))

    elif method_name == 'FFT':
        F = np.fft.fft2(img_f)
        mag = np.abs(F)
        phase = np.angle(F)
        seed_mag = key_data[0] if isinstance(key_data, tuple) else 0.5
        seed_phase = key_data[1] if isinstance(key_data, tuple) else 0.7
        perm_mag = _chaos_perm(mag.size, seed=seed_mag)
        perm_phase = _chaos_perm(phase.size, seed=seed_phase)
        mag_s = mag.ravel()[perm_mag].reshape(mag.shape)
        ph_s = phase.ravel()[perm_phase].reshape(phase.shape)
        result = np.fft.ifft2(mag_s * np.exp(1j * ph_s)).real

    elif method_name == 'Chaos':
        seed = key_data if key_data is not None else 0.5
        perm = _chaos_perm(n, seed=seed)
        permuted = flat[perm]
        key = (np.abs(np.sin(np.arange(n, dtype=np.float64))) * 255).astype(np.uint8)
        result = np.bitwise_xor(permuted, key).reshape(h, w).astype(np.float32)

    elif method_name == 'DNA':
        key_seed = key_data if key_data is not None else None
        result = _dna_encrypt(flat, key_seed).reshape(h, w).astype(np.float32)

    elif method_name == 'ECC':
        seed = key_data if key_data is not None else 0.314159
        perm = _chaos_perm(n, seed=seed)
        result = flat[perm].reshape(h, w).astype(np.float32)

    elif method_name == 'AES':
        master_key = key_data if key_data is not None else None
        result = _aes_encrypt(flat, master_key).reshape(h, w).astype(np.float32)

    elif method_name == 'CellularAutomata':
        result = _ca_mix(flat).reshape(h, w).astype(np.float32)

    elif method_name == 'CompressiveSensing':
        result = _cs_shuffle(flat).reshape(h, w).astype(np.float32)

    elif method_name == 'RubikCube':
        result = _rubik_permute(padded).astype(np.float32)

    else:
        result = img_f

    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return result[:orig_h, :orig_w]

# ------------------------------------------------------------------
# Process encryption (single or hybrid)
# ------------------------------------------------------------------
def _encrypt_channel(channel, method_param, hybrid, method1, method2, key1, key2, key):
    """Encrypt a single 2-D grayscale channel."""
    if hybrid and method1 and method2:
        stage1 = apply_encryption(channel, method1, key_data=key1)
        return apply_encryption(stage1, method2, key_data=key2)
    return apply_encryption(channel, method_param, key_data=key)


def process_encryption(img_path, method_param, hybrid=False, method1=None, method2=None,
                       key_id=None, key1_id=None, key2_id=None):
    # Read in color to preserve the original image's color information
    img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_color is None:
        return None

    key1_entry = get_key(key1_id) if key1_id else None
    key2_entry = get_key(key2_id) if key2_id else None
    key_entry  = get_key(key_id)  if key_id  else None
    key1 = key1_entry['key_data'] if key1_entry else None
    key2 = key2_entry['key_data'] if key2_entry else None
    key  = key_entry['key_data']  if key_entry  else None

    b, g, r = cv2.split(img_color)
    enc_b = _encrypt_channel(b, method_param, hybrid, method1, method2, key1, key2, key)
    enc_g = _encrypt_channel(g, method_param, hybrid, method1, method2, key1, key2, key)
    enc_r = _encrypt_channel(r, method_param, hybrid, method1, method2, key1, key2, key)
    result_img = cv2.merge([enc_b, enc_g, enc_r])

    out_filename = f"enc_{uuid4().hex}.png"
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_filename)
    cv2.imwrite(out_path, result_img)
    return out_filename

# ------------------------------------------------------------------
# Analysis functions
# ------------------------------------------------------------------
def compute_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist.flatten().astype(int).tolist()

def compute_entropy(image):
    pixel_counts = np.bincount(image.flatten(), minlength=256)
    total = image.size
    probs = pixel_counts / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return round(entropy, 4)

def compute_correlation(image):
    h, w = image.shape
    hor_corr = np.corrcoef(image[:, :-1].flatten(), image[:, 1:].flatten())[0, 1]
    ver_corr = np.corrcoef(image[:-1, :].flatten(), image[1:, :].flatten())[0, 1]
    diag_corr = np.corrcoef(image[:-1, :-1].flatten(), image[1:, 1:].flatten())[0, 1]
    return {
        'horizontal': float(hor_corr) if not np.isnan(hor_corr) else 0.0,
        'vertical': float(ver_corr) if not np.isnan(ver_corr) else 0.0,
        'diagonal': float(diag_corr) if not np.isnan(diag_corr) else 0.0
    }

# ==================================================================
# ATTACK FUNCTIONS
# ==================================================================

def _compute_similarity(img1, img2):
    """Compute PSNR and SSIM-like correlation between two images."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        psnr = 100.0
    else:
        psnr = 10 * np.log10(255 ** 2 / mse)
    # Correlation as proxy for SSIM
    corr = np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
    if np.isnan(corr):
        corr = 0.0
    return round(psnr, 3), round(float(corr), 5)

def _differential_score(orig, attacked):
    """
    NPCR (Number of Pixels Change Rate) and UACI (Unified Average Changing Intensity).
    High NPCR (~99.6%) and UACI (~33.4%) = good encryption.
    Low values = vulnerable to differential attack.
    """
    orig = orig.astype(np.float64)
    attacked = attacked.astype(np.float64)
    diff = orig != attacked
    npcr = float(diff.sum()) / diff.size * 100.0
    uaci = float(np.abs(orig - attacked).sum()) / (diff.size * 255) * 100.0
    return round(npcr, 4), round(uaci, 4)

# ------------------------------------------------------------------
# BRUTE FORCE ATTACK
# Attempts a small set of candidate keys / seeds for the given method
# and scores each attempt against the encrypted image.
# For methods with no key (CA, CS, Rubik), it tries parameter tweaks.
# ------------------------------------------------------------------
def brute_force_attack(enc_img, method):
    """
    Simulates a brute force attack by trying candidate keys and
    measuring how close the re-encrypted output is to the input.
    Returns ranked candidates with similarity scores.
    """
    h, w = enc_img.shape
    candidates = []

    # --- Build candidate key space per method ---
    if method in ('Chaos', 'DCT', 'DWT', 'ECC'):
        # Try 20 uniformly-spaced seeds in [0,1]
        seeds = [round(i / 19, 6) for i in range(20)]
        for seed in seeds:
            try:
                attempt = apply_encryption(enc_img, method, key_data=seed)
                psnr, corr = _compute_similarity(enc_img, attempt)
                candidates.append({'key': str(seed), 'psnr': psnr, 'correlation': corr})
            except Exception:
                pass

    elif method == 'AES':
        # Try 10 fixed seeds to generate master keys
        test_seeds = [0, 42, 123, 999, 0xDEADBEEF, 0xCAFEBABE, 7777, 31337, 54321, 11111]
        for s in test_seeds:
            try:
                rng = np.random.default_rng(s)
                mk = rng.integers(0, 256, size=256, dtype=np.uint8)
                attempt = apply_encryption(enc_img, 'AES', key_data=mk)
                psnr, corr = _compute_similarity(enc_img, attempt)
                candidates.append({'key': f'seed={s}', 'psnr': psnr, 'correlation': corr})
            except Exception:
                pass

    elif method == 'DNA':
        # Try integer seeds 0..19
        for s in range(20):
            try:
                attempt = apply_encryption(enc_img, 'DNA', key_data=s)
                psnr, corr = _compute_similarity(enc_img, attempt)
                candidates.append({'key': f'seed={s}', 'psnr': psnr, 'correlation': corr})
            except Exception:
                pass

    elif method == 'FFT':
        pairs = [(0.3, 0.5), (0.5, 0.7), (0.1, 0.9), (0.618, 0.382),
                 (0.25, 0.75), (0.4, 0.6), (0.2, 0.8), (0.55, 0.45),
                 (0.7, 0.3), (0.15, 0.85)]
        for p in pairs:
            try:
                attempt = apply_encryption(enc_img, 'FFT', key_data=p)
                psnr, corr = _compute_similarity(enc_img, attempt)
                candidates.append({'key': str(p), 'psnr': psnr, 'correlation': corr})
            except Exception:
                pass

    else:
        # CellularAutomata, CompressiveSensing, RubikCube – keyless
        # Try re-applying the same method multiple times
        for n_apply in range(1, 6):
            try:
                result = enc_img.copy()
                for _ in range(n_apply):
                    result = apply_encryption(result, method)
                psnr, corr = _compute_similarity(enc_img, result)
                candidates.append({'key': f'repeat×{n_apply}', 'psnr': psnr, 'correlation': corr})
            except Exception:
                pass

    # Sort by PSNR descending (higher = closer match)
    candidates.sort(key=lambda x: x['psnr'], reverse=True)
    best = candidates[0] if candidates else None

    # Resistance score: 100 = fully resistant, 0 = broken
    # If best PSNR > 30 dB, attacker is close → low resistance
    best_psnr = best['psnr'] if best else 0
    if best_psnr >= 30:
        resistance = max(0, round(100 - (best_psnr - 30) * 5, 1))
    else:
        resistance = min(100, round(100 - best_psnr * 1.5, 1))

    # Generate attack result image (best candidate re-encrypted image)
    best_attempt_img = None
    if best and method in ('Chaos', 'DCT', 'DWT', 'ECC'):
        seed = float(best['key'])
        best_attempt_img = apply_encryption(enc_img, method, key_data=seed)
    elif best and method == 'AES':
        s = int(best['key'].split('=')[1])
        rng = np.random.default_rng(s)
        mk = rng.integers(0, 256, size=256, dtype=np.uint8)
        best_attempt_img = apply_encryption(enc_img, method, key_data=mk)
    elif best and method == 'DNA':
        s = int(best['key'].split('=')[1])
        best_attempt_img = apply_encryption(enc_img, method, key_data=s)
    else:
        best_attempt_img = apply_encryption(enc_img, method)

    return {
        'attack_type': 'Brute Force',
        'method': method,
        'candidates_tested': len(candidates),
        'best_candidate': best,
        'top5': candidates[:5],
        'resistance_score': resistance,
        'resistance_label': _resistance_label(resistance),
        'attack_image': best_attempt_img
    }

# ------------------------------------------------------------------
# DIFFERENTIAL ATTACK
# Flips 1 bit in the encrypted image and measures how much the
# re-encryption output changes. Also analyzes pixel difference maps.
# ------------------------------------------------------------------
def differential_attack(enc_img, method):
   
    h, w = enc_img.shape
    results = []
    attack_images = []

    ref_enc = enc_img.copy()

    
    test_positions = [
        (h // 2, w // 2),
        (h // 4, w // 4),
        (3 * h // 4, 3 * w // 4)
    ]

    for py, px in test_positions:
        
        
        modified_input = enc_img.copy()
        modified_input[py, px] = modified_input[py, px] ^ 1 
        
        perturbed_enc = apply_encryption(modified_input, method)
        
        
        npcr, uaci = _differential_score(ref_enc, perturbed_enc)

        diff_map = np.abs(ref_enc.astype(np.int32) - perturbed_enc.astype(np.int32)).astype(np.uint8)
        diff_colored = cv2.applyColorMap(cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_HOT)
        
        results.append({
            'position': [int(py), int(px)],
            'npcr': npcr,
            'uaci': uaci,
        })
        attack_images.append(cv2.cvtColor(diff_colored, cv2.COLOR_BGR2GRAY))

    avg_npcr = round(float(np.mean([r['npcr'] for r in results])), 4)
    avg_uaci = round(float(np.mean([r['uaci'] for r in results])), 4)

    npcr_score = min(100, (avg_npcr / 99.6) * 100)
    uaci_score = min(100, (avg_uaci / 33.4) * 100)
    
    resistance = round((npcr_score * 0.7 + uaci_score * 0.3), 1)

    best_idx = int(np.argmax([np.std(img) for img in attack_images]))
    
    return {
        'attack_type': 'Differential',
        'method': method,
        'per_position': results,
        'avg_npcr': avg_npcr,
        'avg_uaci': avg_uaci,
        'ideal_npcr': 99.6,
        'ideal_uaci': 33.4,
        'resistance_score': resistance,
        'resistance_label': _resistance_label(resistance),
        'attack_image': attack_images[best_idx]
    }
def _resistance_label(score):
    if score >= 85:
        return 'Strong'
    elif score >= 60:
        return 'Moderate'
    elif score >= 35:
        return 'Weak'
    else:
        return 'Vulnerable'

# ------------------------------------------------------------------
# DECRYPTION dispatcher
# ------------------------------------------------------------------
def apply_decryption(img, method_name, key_data=None):
    """
    Reverse of apply_encryption for invertible methods.
    Non-invertible methods (CellularAutomata, CompressiveSensing) cannot be reversed.
    """
    orig_h, orig_w = img.shape
    padded, _, _ = _pad_to_aligned(img)
    h, w = padded.shape
    img_f = padded.astype(np.float32)
    flat = padded.flatten().astype(np.uint8)
    n = len(flat)

    if method_name == 'DCT':
        # Reverse: apply inverse permutation in DCT domain
        dct = cv2.dct(img_f)
        perm = _chaos_perm(dct.size, seed=key_data if key_data is not None else 0.5)
        inv_perm = np.argsort(perm)
        unscr = dct.ravel()[inv_perm].reshape(dct.shape)
        result = cv2.idct(unscr)

    elif method_name == 'DWT':
        seed = key_data if key_data is not None else 0.6
        LL, LH, HL, HH = _haar_dwt_2d(img_f)
        perm = _chaos_perm(LL.size, seed=seed)
        inv_perm = np.argsort(perm)
        def _unscr(b):
            return b.ravel()[inv_perm].reshape(b.shape)
        result = _haar_idwt_2d(_unscr(LL), _unscr(LH), _unscr(HL), _unscr(HH))

    elif method_name == 'FFT':
        F = np.fft.fft2(img_f)
        mag = np.abs(F)
        phase = np.angle(F)
        seed_mag = key_data[0] if isinstance(key_data, tuple) else 0.5
        seed_phase = key_data[1] if isinstance(key_data, tuple) else 0.7
        perm_mag = _chaos_perm(mag.size, seed=seed_mag)
        perm_phase = _chaos_perm(phase.size, seed=seed_phase)
        inv_mag = np.argsort(perm_mag)
        inv_phase = np.argsort(perm_phase)
        mag_u = mag.ravel()[inv_mag].reshape(mag.shape)
        ph_u = phase.ravel()[inv_phase].reshape(phase.shape)
        result = np.fft.ifft2(mag_u * np.exp(1j * ph_u)).real

    elif method_name == 'Chaos':
        # Reverse: undo XOR then undo permutation
        seed = key_data if key_data is not None else 0.5
        perm = _chaos_perm(n, seed=seed)
        inv_perm = np.argsort(perm)
        key = (np.abs(np.sin(np.arange(n, dtype=np.float64))) * 255).astype(np.uint8)
        xored = np.bitwise_xor(flat, key)
        result = xored[inv_perm].reshape(h, w).astype(np.float32)

    elif method_name == 'DNA':
        # DNA XOR is symmetric (same op reverses it)
        key_seed = key_data if key_data is not None else None
        result = _dna_encrypt(flat, key_seed).reshape(h, w).astype(np.float32)

    elif method_name == 'ECC':
        seed = key_data if key_data is not None else 0.314159
        perm = _chaos_perm(n, seed=seed)
        inv_perm = np.argsort(perm)
        result = flat[inv_perm].reshape(h, w).astype(np.float32)

    elif method_name == 'AES':
        # Reverse 5 rounds of AES-like in reverse order
        master_key = key_data if key_data is not None else None
        if master_key is None:
            rng = np.random.default_rng(0xDEADBEEF)
            master_key = rng.integers(0, 256, size=256, dtype=np.uint8)
        inv_sbox = np.argsort(_SBOX).astype(np.uint8)
        data = flat.astype(np.uint8).copy()
        for rnd in reversed(range(5)):
            rk = np.tile(np.roll(master_key, rnd * 37), (n // 256) + 1)[:n].astype(np.uint8)
            data = np.bitwise_xor(data, rk)
            tail = n % 16
            main_n = n - tail
            if main_n > 0:
                block = data[:main_n].reshape(-1, 16)
                block = np.roll(block, (rnd + 1), axis=1)
                data[:main_n] = block.ravel()
            data = inv_sbox[data]
        result = data.reshape(h, w).astype(np.float32)

    elif method_name == 'RubikCube':
        # Reverse Rubik: apply opposite rotations
        bs = _ALIGN
        out = padded.copy().astype(np.float32)
        src = padded
        for bi, i in enumerate(range(0, h, bs)):
            for bj, j in enumerate(range(0, w, bs)):
                ie, je = min(i + bs, h), min(j + bs, w)
                k = (bi + bj) % 4
                if k:
                    out[i:ie, j:je] = np.rot90(src[i:ie, j:je], k=(4 - k))
        result = out

    else:
        result = img_f

    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return result[:orig_h, :orig_w]


def process_decryption(img_path, method_param, key_id=None):
    # Read in color (OpenCV always returns 3-channel BGR from IMREAD_COLOR)
    img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_color is None:
        return None

    key_entry = get_key(key_id) if key_id else None
    key = key_entry['key_data'] if key_entry else None

    b, g, r = cv2.split(img_color)

    # Detect if encrypted file is true-color or a grayscale image OpenCV
    # expanded into 3 identical channels.
    is_grayscale_source = np.array_equal(b, g) and np.array_equal(b, r)

    if is_grayscale_source:
        # Encrypted from a grayscale source -- decrypt one channel, then
        # convert to BGR so it saves as a proper color PNG.
        dec_gray = apply_decryption(b, method_param, key_data=key)
        result_img = cv2.cvtColor(dec_gray, cv2.COLOR_GRAY2BGR)
    else:
        # True-color encrypted image -- decrypt each channel independently.
        dec_b = apply_decryption(b, method_param, key_data=key)
        dec_g = apply_decryption(g, method_param, key_data=key)
        dec_r = apply_decryption(r, method_param, key_data=key)
        result_img = cv2.merge([dec_b, dec_g, dec_r])

    out_filename = f"dec_{uuid4().hex}.png"
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_filename)
    cv2.imwrite(out_path, result_img)
    return out_filename, is_grayscale_source


# --- Decryption endpoint ---
@app.route('/decrypt', methods=['POST'])
def decrypt():
    NON_INVERTIBLE = {'CellularAutomata', 'CompressiveSensing'}
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        method = request.form.get('method', 'DCT')
        key_id = request.form.get('key_id')

        if method in NON_INVERTIBLE:
            return jsonify({"error": f"{method} is a one-way transform and cannot be decrypted."}), 400

        if key_id and not get_key(key_id):
            return jsonify({"error": "Selected key not found. Please re-enter the key."}), 400

        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_dec_{uuid4().hex}.png")
        file.save(temp_path)
        result = process_decryption(temp_path, method, key_id=key_id)
        os.remove(temp_path) if os.path.exists(temp_path) else None

        if result:
            out_file, was_grayscale = result
            resp = {
                "result_url": f"http://127.0.0.1:5000/static/{out_file}",
                "method_used": method
            }
            if was_grayscale:
                resp["warning"] = (
                    "The encrypted image was saved without color information (encrypted with an older version). "
                    "Please re-encrypt your original color image to get a full-color decrypted result."
                )
            return jsonify(resp)
        return jsonify({"error": "Decryption failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/static/<filename>')
def serve_static(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# --- Key Management ---
@app.route('/generate_key', methods=['POST'])
def generate_key():
    try:
        data = request.get_json()
        method = data.get('method')
        seed = data.get('seed')
        raw_key = data.get('raw_key')

        if method not in ['AES', 'Chaos']:
            return jsonify({'error': 'Only AES and Chaos key generation supported'}), 400

        if raw_key is not None:
            if method == 'AES':
                h = hashlib.sha256(raw_key.encode()).hexdigest()
                seed_int = int(h, 16)
                rng = get_rng(seed_int)
                key_data = rng.integers(0, 256, size=256, dtype=np.uint8)
            elif method == 'Chaos':
                try:
                    key_data = float(raw_key)
                except (ValueError, TypeError):
                    return jsonify({'error': 'Chaos key must be a number (e.g., 0.123456)'}), 400
        else:
            if method == 'AES':
                rng = get_rng(seed)
                key_data = rng.integers(0, 256, size=256, dtype=np.uint8)
            elif method == 'Chaos':
                if seed is not None:
                    try:
                        key_data = float(seed)
                    except (ValueError, TypeError):
                        return jsonify({'error': 'Chaos seed must be a number'}), 400
                else:
                    key_data = float(np.random.rand())
            else:
                return jsonify({'error': 'Unsupported method'}), 400

        key_id, fingerprint = store_key(method, key_data)
        return jsonify({'key_id': key_id, 'fingerprint': fingerprint, 'method': method})
    except Exception as e:
        print(f"ERROR in generate_key: {e}")
        return jsonify({'error': f'Key generation failed: {str(e)}'}), 500
    
@app.route('/keys', methods=['GET'])
def list_keys():
    keys = [{'key_id': k, 'method': v['method'], 'fingerprint': v['fingerprint']}
            for k, v in key_store.items()]
    return jsonify(keys)

# --- Encryption ---
@app.route('/encrypt', methods=['POST'])
def encrypt():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        hybrid = request.form.get('hybrid') == 'true'
        method = request.form.get('method')
        method1 = request.form.get('method1')
        method2 = request.form.get('method2')
        key_id = request.form.get('key_id')
        key1_id = request.form.get('key1_id')
        key2_id = request.form.get('key2_id')

        if not hybrid and key_id and not get_key(key_id):
            return jsonify({"error": "Selected key not found. Please regenerate."}), 400
        if hybrid:
            if (key1_id and not get_key(key1_id)) or (key2_id and not get_key(key2_id)):
                return jsonify({"error": "One of the selected keys not found."}), 400

        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid4().hex}.png")
        file.save(temp_path)

        if hybrid and method1 and method2:
            out_file = process_encryption(temp_path, None, hybrid=True,
                                          method1=method1, method2=method2,
                                          key1_id=key1_id, key2_id=key2_id)
            used_method = f"Hybrid: {method1} → {method2}"
        else:
            out_file = process_encryption(temp_path, method, key_id=key_id)
            used_method = method

        os.remove(temp_path) if os.path.exists(temp_path) else None

        if out_file:
            return jsonify({
                "result_url": f"http://127.0.0.1:5000/static/{out_file}",
                "method_used": used_method
            })
        return jsonify({"error": "Encryption failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Analysis endpoint ---
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image file"}), 400
    file = request.files['image']
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"analyze_{uuid4().hex}.png")
    file.save(temp_path)
    img = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
    os.remove(temp_path) if os.path.exists(temp_path) else None
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    hist = compute_histogram(img)
    entropy = compute_entropy(img)
    corr = compute_correlation(img)
    return jsonify({
        "histogram": hist,
        "entropy": entropy,
        "correlation": corr
    })

# ==================================================================
# ATTACK ENDPOINTS
# ==================================================================

@app.route('/attack/brute_force', methods=['POST'])
def attack_brute_force():
    """
    Accepts an encrypted image + method name.
    Runs brute force key search and returns resistance metrics + attack image.
    """
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file"}), 400
        file = request.files['image']
        method = request.form.get('method', 'Chaos')

        valid_methods = ['DCT', 'DWT', 'FFT', 'Chaos', 'DNA', 'ECC', 'AES',
                         'CellularAutomata', 'CompressiveSensing', 'RubikCube']
        if method not in valid_methods:
            return jsonify({"error": f"Unsupported method: {method}"}), 400

        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"atk_{uuid4().hex}.png")
        file.save(temp_path)
        enc_img = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
        os.remove(temp_path) if os.path.exists(temp_path) else None

        if enc_img is None:
            return jsonify({"error": "Invalid image"}), 400

        t0 = time.time()
        result = brute_force_attack(enc_img, method)
        elapsed = round(time.time() - t0, 3)

        # Save attack image
        atk_img = result.pop('attack_image')
        atk_filename = f"atk_bf_{uuid4().hex}.png"
        atk_path = os.path.join(app.config['UPLOAD_FOLDER'], atk_filename)
        if atk_img is not None:
            cv2.imwrite(atk_path, atk_img)
            result['attack_image_url'] = f"http://127.0.0.1:5000/static/{atk_filename}"
        else:
            result['attack_image_url'] = None

        result['elapsed_seconds'] = elapsed
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/attack/differential', methods=['POST'])
def attack_differential():
    """
    Accepts an encrypted image + method name.
    Runs differential attack (NPCR/UACI analysis) and returns metrics + diff image.
    """
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file"}), 400
        file = request.files['image']
        method = request.form.get('method', 'Chaos')

        valid_methods = ['DCT', 'DWT', 'FFT', 'Chaos', 'DNA', 'ECC', 'AES',
                         'CellularAutomata', 'CompressiveSensing', 'RubikCube']
        if method not in valid_methods:
            return jsonify({"error": f"Unsupported method: {method}"}), 400

        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"atk_{uuid4().hex}.png")
        file.save(temp_path)
        enc_img = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
        os.remove(temp_path) if os.path.exists(temp_path) else None

        if enc_img is None:
            return jsonify({"error": "Invalid image"}), 400

        t0 = time.time()
        result = differential_attack(enc_img, method)
        elapsed = round(time.time() - t0, 3)

        # Save attack diff image
        atk_img = result.pop('attack_image')
        atk_filename = f"atk_diff_{uuid4().hex}.png"
        atk_path = os.path.join(app.config['UPLOAD_FOLDER'], atk_filename)
        if atk_img is not None:
            cv2.imwrite(atk_path, atk_img)
            result['attack_image_url'] = f"http://127.0.0.1:5000/static/{atk_filename}"
        else:
            result['attack_image_url'] = None

        result['elapsed_seconds'] = elapsed
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/attack/compare_all', methods=['POST'])
def attack_compare_all():
    """
    Runs both attacks on the encrypted image for ALL methods and returns
    a comparison table of resistance scores.
    """
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file"}), 400
        file = request.files['image']
        method = request.form.get('method', 'Chaos')

        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"atk_{uuid4().hex}.png")
        file.save(temp_path)
        enc_img = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
        os.remove(temp_path) if os.path.exists(temp_path) else None

        if enc_img is None:
            return jsonify({"error": "Invalid image"}), 400

        bf = brute_force_attack(enc_img, method)
        bf.pop('attack_image', None)

        df = differential_attack(enc_img, method)
        df.pop('attack_image', None)

        overall = round((bf['resistance_score'] + df['resistance_score']) / 2, 1)

        return jsonify({
            'method': method,
            'brute_force': bf,
            'differential': df,
            'overall_resistance': overall,
            'overall_label': _resistance_label(overall)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 CipherMatrix Server (Encryption + Decryption + Analysis + Attacks)")
    print("📡 http://127.0.0.1:5000")
    print("🔓 Decryption endpoint: /decrypt")
    print("⚔️  Attack endpoints: /attack/brute_force, /attack/differential, /attack/compare_all")
    print("=" * 60)
app.run(debug=False)