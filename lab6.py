"""
Лабораторна робота 6 — JPEG-lite
Розробка мультимедійних застосунків | Python + NumPy

Запуск:
    python lab6.py
    python lab6.py шлях/до/зображення.png

Якщо зображення не передано, використовується тестове grayscale-зображення
camera з scikit-image. Скрипт створює curve.png, q95.png, q90.png,
q75.png, q50.png, q30.png, q15.png, q05.png та results.csv.
"""
from pathlib import Path
import sys
import csv
import zlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from skimage import data

# Базова таблиця квантування яскравості JPEG для якості 50.
Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99],
], dtype=np.float64)

QUALITIES = [95, 90, 75, 50, 30, 15, 5]
RAW_SIZE = 512 * 512

def dct_matrix(N=8):
    """Створює ортонормовану матрицю базису DCT."""
    D = np.zeros((N, N), dtype=np.float64)
    for k in range(N):
        a = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
        for n in range(N):
            D[k, n] = a * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    return D

D = dct_matrix(8)

def dct2(block):
    """Двовимірне пряме DCT для блока 8×8."""
    return D @ block @ D.T

def idct2(coeffs):
    """Двовимірне зворотне DCT для блока 8×8."""
    return D.T @ coeffs @ D

def qtable(q):
    """Повертає таблицю квантування для якості 1..99."""
    q = max(1, min(99, int(q)))
    S = 5000 / q if q < 50 else 200 - 2 * q
    return np.clip(np.floor((S * Q50 + 50) / 100), 1, 255)

def blockify(img):
    """Розбиває 2D-зображення на блоки 8×8."""
    h, w = img.shape
    if h % 8 or w % 8:
        raise ValueError("Розміри зображення мають бути кратні 8.")
    return np.array([
        img[y:y+8, x:x+8]
        for y in range(0, h, 8)
        for x in range(0, w, 8)
    ])

def unblockify(blocks, h, w):
    """Збирає блоки 8×8 у зображення початкового розміру."""
    result = np.zeros((h, w), dtype=np.float64)
    idx = 0
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            result[y:y+8, x:x+8] = blocks[idx]
            idx += 1
    return result

def encode(img, q):
    """Кодує зображення: блоки, DCT, квантування."""
    Q = qtable(q)
    blocks = blockify(img.astype(np.float64) - 128)
    coeffs = np.array([np.round(dct2(b) / Q) for b in blocks])
    return coeffs.astype(np.int16), Q

def decode(coeffs, Q, h, w):
    """Декодує коефіцієнти та відновлює зображення."""
    blocks = np.array([idct2(c * Q) for c in coeffs])
    img = unblockify(blocks, h, w) + 128
    return np.clip(img, 0, 255).round().astype(np.uint8)

def psnr(original, reconstructed):
    """Обчислює PSNR у децибелах."""
    a = original.astype(np.float64)
    b = reconstructed.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10((255 ** 2) / mse))

def ssim(original, reconstructed):
    """Обчислює SSIM у діапазоні від 0 до 1."""
    return float(structural_similarity(original, reconstructed, data_range=255))

def compressed_size(coeffs):
    """Оцінює розмір коефіцієнтів після пакування zlib."""
    return len(zlib.compress(coeffs.tobytes(), level=9))

def load_gray(path=None):
    """Завантажує зображення у відтінках сірого та обрізає до кратних 8 розмірів."""
    if path:
        img = Image.open(path).convert("L")
        arr = np.asarray(img, dtype=np.uint8)
    else:
        # Вбудоване тестове зображення; точні контрольні значення PDF
        # можуть відрізнятися, оскільки вихідна фотографія інша.
        arr = np.asarray(data.camera(), dtype=np.uint8)
    h = (arr.shape[0] // 8) * 8
    w = (arr.shape[1] // 8) * 8
    if h == 0 or w == 0:
        raise ValueError("Зображення замале для блоків 8×8.")
    return arr[:h, :w].copy()

def run_tests():
    """Базові перевірки матриці DCT, квантування та кодування."""
    assert D.shape == (8, 8)
    assert np.allclose(D @ D.T, np.eye(8), atol=1e-12), "Рядки DCT не ортонормовані"
    sample = np.arange(64, dtype=np.float64).reshape(8, 8)
    assert np.allclose(idct2(dct2(sample)), sample, atol=1e-12), "DCT/IDCT не оборотні"
    assert np.array_equal(qtable(50), Q50), "qtable(50) не збігається з Q50"
    assert qtable(15).min() >= 1 and qtable(15).max() <= 255
    # Контрольний блок із завдання: перевіряємо, що квантування дає цілі коефіцієнти.
    coeffs, Q = encode(np.full((8, 8), 128, dtype=np.uint8), 50)
    assert coeffs.shape == (1, 8, 8)
    print("Тести: OK — DCT, IDCT, таблиця квантування та кодер.")

def evaluate(img, qualities=QUALITIES):
    """Повертає таблицю вимірювань і збережені реконструкції."""
    rows = []
    reconstructions = {}
    raw_nbytes = img.nbytes
    for q in qualities:
        coeffs, Q = encode(img, q)
        rec = decode(coeffs, Q, *img.shape)
        reconstructions[q] = rec
        size = compressed_size(coeffs)
        rows.append({
            "q": q,
            "bytes": size,
            "compression": raw_nbytes / max(size, 1),
            "zeros_percent": 100 * np.count_nonzero(coeffs == 0) / coeffs.size,
            "PSNR": psnr(img, rec),
            "SSIM": ssim(img, rec),
        })
        Image.fromarray(rec).save(f"q{q:02d}.png")
    return rows, reconstructions

def save_results(rows):
    with open("results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def plot_curve(rows):
    qs = [r["q"] for r in rows]
    sizes = [r["bytes"] / 1024 for r in rows]
    psnrs = [r["PSNR"] for r in rows]
    ssims = [r["SSIM"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(qs, sizes, "o-", label="Розмір, КБ")
    axes[0].set_xlabel("Якість q")
    axes[0].set_ylabel("Розмір, КБ")
    axes[0].set_title("Розмір файлу")
    axes[0].grid(True, alpha=.3)
    axes[1].plot(qs, psnrs, "o-", label="PSNR, дБ")
    axes[1].set_xlabel("Якість q")
    axes[1].set_ylabel("PSNR, дБ")
    axes[1].set_title("Якість відновлення")
    axes[1].grid(True, alpha=.3)
    fig.suptitle("JPEG-lite: розмір і якість")
    fig.tight_layout()
    fig.savefig("curve.png", dpi=160)
    plt.close(fig)

def main():
    run_tests()
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    img = load_gray(img_path)
    rows, reconstructions = evaluate(img)
    save_results(rows)
    plot_curve(rows)
    print(f"Зображення: {img.shape[1]}×{img.shape[0]}, сирий розмір: {img.nbytes:,} байт")
    print(" q | байт | стиснення | нулів, % | PSNR, дБ | SSIM")
    for r in rows:
        print(f'{r["q"]:2d} | {r["bytes"]:6d} | {r["compression"]:8.2f}x | '
              f'{r["zeros_percent"]:8.2f} | {r["PSNR"]:8.2f} | {r["SSIM"]:.4f}')
    print("Створено curve.png, q95.png, q90.png, q75.png, q50.png, q30.png, q15.png, q05.png і results.csv.")
    print("Для власного порога видимості відкрийте оригінал і q*.png у масштабі 100%.")

if __name__ == "__main__":
    main()
