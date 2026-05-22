# CipherMatrix – Image Encryption & Security Analysis Platform

## Overview
CipherMatrix is a full-stack image encryption system designed to secure digital images using advanced cryptographic techniques while providing real-time analysis of encryption strength. The platform combines multiple encryption algorithms, attack simulations, and statistical validation tools into a single interactive environment.

---

## Features

### Multi-Algorithm Hybrid Encryption
- Combines 10 encryption techniques:
  - DCT (Discrete Cosine Transform)
  - DWT (Discrete Wavelet Transform)
  - FFT (Fast Fourier Transform)
  - Chaos Logistic Map
  - DNA Encoding
  - AES (Advanced Encryption Standard)
  - ECC-inspired permutation
  - Cellular Automata
  - Compressive Sensing
  - Rubik’s Cube block transformation
- Supports hybrid chaining for layered security

### Real-Time Cryptographic Analytics
- Entropy calculation (ideal ≈ 8.0)
- NPCR & UACI for differential attack resistance
- Pixel correlation analysis (horizontal, vertical, diagonal)
- Histogram visualization for randomness validation

### Attack Simulation Lab
- Brute-force attack simulation
- Differential attack testing
- Resistance scoring system (Strong / Moderate / Weak / Vulnerable)

### Key & Vault Management
- Secure in-memory key storage
- Key fingerprinting and validation
- Passphrase-based key derivation

### Lossless Decryption
- MSE = 0 (no data loss)
- SSIM = 1.0 (perfect reconstruction)

### Interactive UI
- Drag-and-drop image upload
- Real-time processing and visualization
- Responsive design

---

## Tech Stack

### Frontend
- HTML5
- TailwindCSS
- JavaScript
- Chart.js
- Three.js

### Backend
- Python 3
- Flask
- NumPy
- OpenCV

### Architecture
- RESTful API
- Modular and scalable design

---

## Installation & Setup

### Prerequisites
- Python 3.x
- pip

### Run Locally
```bash
# Clone the repository
git clone https://github.com/Rodyyyyy/CipherMatrix.git
cd CipherMatrix

# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
