# Backpropagation & Perceptron — Kecerdasan Buatan

Nama : Mohammad Ferdian Samputra  
NIM  : H1D024023

Implementasi dua algoritma machine learning klasik untuk klasifikasi bipolar menggunakan Python dan NumPy.

---

## Deskripsi Umum

| Algoritma | Masalah | Jenis Data |
|---|---|---|
| Backpropagation | XOR | Non-linearly separable |
| Perceptron | OR | Linearly separable |

Kedua program menggunakan representasi **bipolar** untuk input dan target (nilai -1 dan 1).

---


## Program 1 & 2 — Backpropagation (XOR)

### Cara Menjalankan (`Backpropagation_xor.py`)

```python
import numpy as np
import Backpropagation as b

X = np.array([[1,1], [1,-1], [-1,1], [-1,-1]])
t = np.array([[-1], [1], [1], [-1]])

model = b.Backpropagation(alpha=0.3, epoch=1000, target_error=0.001)
model.fit(X, t)
```

| Parameter | Nilai | Keterangan |
|---|---|---|
| `alpha` | 0.3 | Learning rate |
| `epoch` | 1000 | Maksimal iterasi |
| `target_error` | 0.001 | Batas SSE untuk berhenti |

---

### Fungsi-Fungsi dalam `Backpropagation.py`

#### `__init__(self, alpha, epoch, target_error)`
Konstruktor kelas. Menyimpan hyperparameter dan menginisialisasi bobot serta bias secara acak menggunakan `np.random.rand`.

| Atribut | Shape | Keterangan |
|---|---|---|
| `w_hidden` | (2, 2) | Bobot hidden layer |
| `b_hidden` | (1, 2) | Bias hidden layer |
| `w_output` | (2, 1) | Bobot output layer |
| `b_output` | (1, 1) | Bias output layer |

---

#### `bi_sigmoid(self, x)`
Menerapkan fungsi aktivasi **sigmoid bipolar (tanh)** pada input `x`.

```
f(x) = tanh(x)
```

Output bernilai antara **-1 hingga 1**.

---

#### `deriv_bi_sigmoid(self, x)`
Menghitung **turunan fungsi tanh**, dengan asumsi `x` adalah output dari `bi_sigmoid`.

```
f'(x) = 1 - x²
```

Digunakan saat Backward Propagation untuk menghitung nilai delta.

---

#### `plot_error(self, x, epoch)`
Membuat grafik konvergensi **Sum Square Error (SSE)** selama pelatihan.

- Sumbu X → Epoch
- Sumbu Y → Nilai SSE
- Menampilkan anotasi pada titik akhir pelatihan (epoch dan nilai error terakhir)

---

#### `fit(self, X, t)`
Fungsi utama yang menjalankan keseluruhan proses pelatihan Backpropagation. Hasil log disimpan ke `hasilBackpropagation.txt`.

**Alur per epoch:**

```
Untuk setiap data (xi, target):
│
├── Forward Propagation
│   ├── h_in = xi · w_hidden + b_hidden
│   ├── h    = tanh(h_in)
│   ├── y_in = h · w_output + b_output
│   └── y    = tanh(y_in)
│
├── Backward Propagation
│   ├── error   = target - y
│   ├── d_y     = error × f'(y)
│   ├── error_h = d_y · w_output^T
│   └── d_h     = error_h × f'(h)
│
└── Update Bobot & Bias
    ├── w_output += h^T · d_y × alpha
    ├── b_output += sum(d_y) × alpha
    ├── w_hidden += xi^T · d_h × alpha
    └── b_hidden += sum(d_h) × alpha

Setelah semua data diproses:
├── Hitung SSE = total_error / jumlah_data
└── Berhenti jika SSE < target_error atau epoch maksimal tercapai
```

---

## Program 3 & 4 — Perceptron (OR)

### Cara Menjalankan (`Perceptron_or.py`)

```python
import numpy as np
import Perceptron as p

X = np.array([[1,1], [1,-1], [-1,1], [-1,-1]])
t = np.array([[1], [1], [1], [-1]])

model = p.Perceptron(alpha=0.1, epoch=10)
model.fit(X, t)
```

| Parameter | Nilai | Keterangan |
|---|---|---|
| `alpha` | 0.1 | Learning rate |
| `epoch` | 10 | Maksimal iterasi |

---

### Fungsi-Fungsi dalam `Perceptron.py`

#### `__init__(self, alpha, epoch)`
Konstruktor kelas. Menyimpan learning rate dan jumlah maksimal epoch.

| Atribut | Default | Keterangan |
|---|---|---|
| `alpha` | 0.1 | Learning rate |
| `epoch` | 10 | Maksimal iterasi |

---

#### `weighted_sum(self, X)`
Menghitung nilai **net input (y_in)**.

```
y_in = X · w + b
```

- `self.w_[1:]` → Bobot
- `self.w_[0]` → Bias

---

#### `predict(self, X)`
Menerapkan **fungsi aktivasi bipolar** pada hasil `weighted_sum`.

```
y =  1  jika y_in >= 0
y = -1  jika y_in < 0
```

---

#### `plot_decision_boundary(self, X, t, epoch)`
Membuat visualisasi **garis pemisah (decision boundary)** pada setiap epoch.

- Menampilkan titik data input dengan warna berdasarkan kelas target
- Menggambar garis pemisah berdasarkan nilai bobot dan bias saat ini

---

#### `fit(self, X, t)`
Fungsi utama yang menjalankan keseluruhan proses pelatihan Perceptron. Hasil log disimpan ke `HasilPerceptron.txt`.

**Alur per epoch:**

```
Inisialisasi: w = 0, b = 0

Untuk setiap data (xi, target):
│
├── y_pred = predict(xi)
├── error  = target - y_pred
└── Update dengan Delta Rule:
    ├── w += alpha × error × xi
    └── b += alpha × error

Setelah semua data diproses:
├── Tampilkan decision boundary
├── Hitung SSE = sum(error²)
└── Berhenti jika SSE = 0 atau epoch maksimal tercapai
```

---

## Output yang Dihasilkan

| Program | File Log | Grafik |
|---|---|---|
| Backpropagation | `hasilBackpropagation.txt` | Grafik konvergensi SSE per epoch |
| Perceptron | `HasilPerceptron.txt` | Grafik decision boundary per epoch |

Isi file log mencakup: bobot awal, proses tiap data per epoch, nilai SSE, bobot akhir, dan kondisi berhenti.
