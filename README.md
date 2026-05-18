[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Xgva0T8K)
# CSC4007 — Lab 4: LSTM/GRU & Long-range Dependencies

## Giới thiệu bài thực hành

Lab 4 nối tiếp tinh thần của Lab 3. Ở Lab 3, sinh viên đã chuyển từ biểu diễn BoW/TF-IDF sang mô hình chuỗi với kiến trúc **Embedding + RNN** và làm quen với việc log thí nghiệm bằng **Weights & Biases (W&B)**.

Tuy nhiên, RNN cơ bản thường gặp khó khăn khi văn bản dài hoặc khi tín hiệu quan trọng nằm cách xa nhau trong câu/review. Lab 4 tập trung vào **LSTM/GRU**, tức các kiến trúc recurrent có cơ chế “gates” để hỗ trợ học phụ thuộc dài hạn tốt hơn.

Trong starter kit này, giảng viên chọn một cấu trúc cơ bản, cố định và dễ hiểu làm điểm xuất phát:

```text
Tokenized text → Embedding → 1-layer LSTM → Dropout → Linear classifier
```

Sinh viên bắt buộc chạy được cấu trúc cơ bản này trước, sau đó phát triển lên các biến thể có kết quả tốt hơn, ví dụ:

- chuyển từ LSTM sang GRU;
- bật `--bidirectional`;
- tăng `hidden_dim` hoặc `embed_dim`;
- thử `num_layers = 2`;
- tinh chỉnh `max_len`, `dropout`, `lr`, `batch_size`;
- phân tích learning curves và lỗi dự đoán sai để giải thích vì sao mô hình tốt hơn hoặc kém hơn.

Mục tiêu quan trọng không chỉ là “accuracy cao hơn”, mà là biết **so sánh công bằng** giữa các mô hình: cùng dataset, cùng split, cùng seed, cùng metric, và có bảng ablation rõ ràng.

---

## Mục tiêu bài thực hành

Sau bài lab này, sinh viên cần:

1. Giải thích được trực giác cơ bản của LSTM/GRU trong xử lý phụ thuộc dài hạn.
2. Chạy được baseline **Embedding + LSTM** cho bài toán phân loại cảm xúc IMDB.
3. Thử ít nhất **2 biến thể nâng cấp** so với baseline.
4. So sánh mô hình bằng `accuracy`, `macro-F1`, confusion matrix và learning curves.
5. Ghi lại cấu hình, kết quả và nhận xét trong báo cáo.
6. Phân tích lỗi trên ít nhất 10 mẫu dự đoán sai.
7. Lưu checkpoint mô hình tốt nhất và các artefact cần thiết trong thư mục `outputs/`.

---

## Dataset

Mặc định bài lab sử dụng **IMDB movie reviews** với 2 nhãn:

- `negative`
- `positive`

Repo hỗ trợ hai chế độ:

1. `--dataset imdb`: tải IMDB qua Hugging Face `datasets`.
2. `--dataset local_csv`: chạy nhanh bằng file CSV nhỏ trong `data/raw/sample_imdb_tiny.csv`, dùng cho smoke test và CI.

Khi dùng IMDB thật:

- giữ nguyên `train` và `test` split gốc;
- tách `validation` từ `train`;
- chỉ dùng `test` để đánh giá cuối cùng;
- nếu truyền `--max_rows`, chỉ lấy một phần nhỏ để kiểm thử nhanh.

---

## Cấu trúc repo

```text
csc4007_lab4_starter_kit/
├── .github/workflows/
│   ├── ci.yml
│   └── imdb-smoke.yml
├── data/
│   └── raw/
│       ├── README.md
│       └── sample_imdb_tiny.csv
├── notebooks/
│   └── README.md
├── outputs/
│   ├── error_analysis/
│   ├── figures/
│   ├── logs/
│   ├── metrics/
│   ├── models/
│   ├── predictions/
│   └── splits/
├── reports/
│   ├── analysis_report.md
│   └── rubric.md
├── requirements.txt
├── run_lab4.py
└── src/
    ├── __init__.py
    ├── data.py
    ├── error_analysis.py
    ├── evaluate.py
    ├── model.py
    ├── sequence_audit.py
    ├── train.py
    ├── utils.py
    └── wandb_utils.py
```

---

## Cài đặt môi trường

```bash
conda create -n csc4007-nlp python=3.10 -y
conda activate csc4007-nlp
pip install -r requirements.txt
```

Nếu dùng `venv`:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Nếu dùng W&B online:

```bash
wandb login
```

---

## Chạy baseline bắt buộc: Embedding + 1-layer LSTM

```bash
python run_lab4.py   --dataset imdb   --seed 42   --model_type lstm   --vocab_size 20000   --max_len 256   --embed_dim 128   --hidden_dim 128   --num_layers 1   --dropout 0.3   --batch_size 64   --epochs 6   --lr 1e-3   --use_wandb
```

Đây là cấu trúc cơ bản giảng viên chọn làm điểm xuất phát. Sinh viên **không nên vội nâng cấp** trước khi chạy được baseline và đọc được các output.

---

## Chạy nhanh bằng dữ liệu nhỏ

```bash
python run_lab4.py   --dataset local_csv   --data_path data/raw/sample_imdb_tiny.csv   --model_type lstm   --epochs 2   --batch_size 4   --max_len 64   --vocab_size 1000   --embed_dim 32   --hidden_dim 32   --wandb_mode disabled
```

Lệnh này dùng để kiểm tra code chạy được, không dùng để kết luận chất lượng mô hình.

---

## Gợi ý nâng cấp để đạt accuracy cao hơn

Sau khi chạy baseline, sinh viên cần thử ít nhất 2 biến thể. Một số cấu hình gợi ý:

### Biến thể 1 — GRU

```bash
python run_lab4.py   --dataset imdb   --seed 42   --model_type gru   --max_len 256   --embed_dim 128   --hidden_dim 128   --dropout 0.3   --epochs 6   --lr 1e-3   --run_name gru_baseline   --use_wandb
```

### Biến thể 2 — BiLSTM

```bash
python run_lab4.py   --dataset imdb   --seed 42   --model_type lstm   --bidirectional   --max_len 256   --embed_dim 128   --hidden_dim 128   --dropout 0.4   --epochs 6   --lr 1e-3   --run_name bilstm   --use_wandb
```

### Biến thể 3 — Stacked GRU/LSTM

```bash
python run_lab4.py   --dataset imdb   --seed 42   --model_type gru   --num_layers 2   --bidirectional   --max_len 256   --embed_dim 128   --hidden_dim 128   --dropout 0.4   --epochs 8   --lr 8e-4   --run_name stacked_bigru   --use_wandb
```

Sinh viên cần ghi lại kết quả vào `reports/analysis_report.md`, không chỉ nộp code.

---

## Output sau khi chạy

Repo sẽ sinh các artefact sau:

```text
outputs/
├── error_analysis/
│   ├── error_analysis.csv
│   └── error_analysis_summary.md
├── figures/
│   ├── confusion_matrix.png
│   ├── loss_curve.png
│   └── metric_curve.png
├── logs/
│   ├── run_config.json
│   ├── run_summary.json
│   └── sequence_audit.md
├── metrics/
│   ├── epoch_history.csv
│   ├── metrics_summary.json
│   ├── metrics_summary.md
│   └── baseline_vs_lab4.csv
├── models/
│   └── best_model.pt
├── predictions/
│   └── test_predictions.csv
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

---

## Yêu cầu sinh viên phải hoàn thành

1. Fork repo starter kit về GitHub cá nhân.
2. Chạy thành công baseline **Embedding + LSTM**.
3. Thử ít nhất **2 biến thể nâng cấp**.
4. Lập bảng ablation gồm tối thiểu các cột: `model_type`, `bidirectional`, `num_layers`, `max_len`, `hidden_dim`, `dropout`, `accuracy`, `macro_f1`, nhận xét ngắn.
5. Phân tích ít nhất 10 mẫu sai trong `outputs/error_analysis/error_analysis.csv`.
6. Điền đầy đủ `reports/analysis_report.md`.
7. Nộp link GitHub repo cá nhân.
8. Nếu dùng W&B online, ghi rõ link project/run trong báo cáo.

---

## Tiêu chí chọn mô hình tốt nhất

Mô hình tốt nhất không nhất thiết là mô hình có loss train thấp nhất. Sinh viên nên ưu tiên:

1. `val_macro_f1` cao và ổn định;
2. khoảng cách train/validation không quá lớn;
3. confusion matrix cân bằng hơn;
4. error analysis cho thấy lỗi giảm ở các mẫu phủ định, chuyển ý, câu dài;
5. test performance tốt sau khi chọn mô hình bằng validation set.

---

## Lỗi thường gặp

- Nhầm shape đầu vào của LSTM/GRU.
- Quên padding hoặc không xử lý độ dài chuỗi.
- So sánh mô hình không công bằng vì dùng split khác nhau.
- Chọn mô hình lớn nhưng không kiểm soát overfitting.
- Chỉ báo cáo accuracy, không báo cáo macro-F1.
- Không lưu checkpoint tốt nhất.
- Không ghi lại cấu hình thí nghiệm.
- Kết luận “mô hình tốt hơn” nhưng không có ablation table.

---

## CI dùng để làm gì?

Repo có 2 workflow:

- `lab4-ci`: chạy smoke test với dữ liệu local nhỏ.
- `lab4-imdb-smoke`: chạy kiểm tra nhanh với IMDB thật, chỉ chạy thủ công qua `workflow_dispatch` để tránh tốn thời gian ở mỗi lần push.

CI chỉ kiểm tra repo có chạy được và sinh artefact cơ bản. CI **không thay thế** cho báo cáo phân tích học thuật.
