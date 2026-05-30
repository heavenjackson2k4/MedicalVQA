# 🏥 Medical VQA — Hỏi Đáp Hình Ảnh Y Tế bằng Vision-Language Models

> **Đồ án cuối kỳ môn Xử Lý Ngôn Ngữ Tự Nhiên — IUH 2025–2026 | Võ Trường Thiên (MSSV: 22676901)**

Hệ thống trả lời câu hỏi y tế dựa trên hình ảnh (X-quang, MRI, CT scan) sử dụng các mô hình thị giác-ngôn ngữ (Vision-Language Models) được fine-tune chuyên biệt cho lĩnh vực y sinh.

---

## 📋 Tổng quan

Medical VQA là bài toán Multimodal AI: nhận đầu vào là **ảnh y tế** + **câu hỏi ngôn ngữ tự nhiên**, sinh ra câu trả lời ngắn gọn chính xác về mặt y khoa.

Dự án đề xuất và so sánh **hai kiến trúc** trên bộ dữ liệu [SLAKE-vqa-english](https://huggingface.co/datasets/mdwiratathya/SLAKE-vqa-english):

| Kiến trúc | Vision Encoder | Language Model | Cầu nối |
|---|---|---|---|
| **Hướng 1** | BiomedCLIP ViT-B/16 (frozen) | Qwen2.5-0.5B + LoRA | MLP Projector (512→1024→896) |
| **Hướng 2** | SigLIP (frozen) | PaliGemma 3B + LoRA | Linear Projector (tích hợp sẵn) |

---

## 🏆 Kết quả trên SLAKE Test Set (n = 1.061)

| Độ đo | BiomedCLIP + Qwen2.5 | PaliGemma 3B |
|---|---|---|
| **Exact Match** | **82,56%** | 76,91% |
| **Token F1** | **85,77%** | 81,23% |
| **BLEU-1** | **85,23%** | 80,41% |
| **BLEU-4** | **19,52%** | 16,43% |
| **ROUGE-L** | **85,61%** | 80,89% |
| **Cosine Similarity** | **93,61%** | 91,32% |
| EM — CLOSED (Yes/No) | **87,89%** | 85,35% |
| EM — OPEN | **79,89%** | 72,66% |

> Phần cứng: Kaggle T4 GPU 16GB VRAM — gói miễn phí

---

## 🏗️ Kiến trúc mô hình

### Hướng 1 — BiomedCLIP + Qwen2.5-0.5B

```
Ảnh y tế (224×224)
        ↓
BiomedCLIP ViT-B/16  ──→  vector 512 chiều  ──→  MLP Projector (512→1024→896)
                                                              ↓
Câu hỏi (ChatML format)  ──→  Qwen Tokenizer  ──→  Embeddings (L × 896)
                                                              ↓
                                              [visual_token | text_tokens]
                                                              ↓
                                              Qwen2.5-0.5B + LoRA  →  Câu trả lời
```

**Điểm thiết kế chính:**
- BiomedCLIP pretrain trên 15M+ cặp ảnh-văn bản từ PubMed — đặc trưng y tế chuyên sâu
- MLP Projector **fully trainable**: học căn chỉnh không gian ảnh (512d) ↔ ngôn ngữ (896d)
- LoRA (r=16, α=32) áp dụng trên `q/k/v/o_proj` của Qwen2.5
- Label masking: chỉ tính cross-entropy loss trên token câu trả lời, không tính phần prompt

### Hướng 2 — PaliGemma 3B + LoRA

```
Ảnh y tế (224×224) + Câu hỏi
        ↓
PaliGemmaProcessor (xử lý hợp nhất)
        ↓
SigLIP Encoder  →  Linear Projector (frozen)  →  Gemma Decoder + LoRA  →  Câu trả lời
```

**Điểm thiết kế chính:**
- VLM thống nhất end-to-end — không cần thiết kế projector thủ công
- Quantization 4-bit NF4 (bitsandbytes) để vừa 16GB VRAM
- LoRA áp dụng toàn bộ model (`q/k/v/o_proj`)

---

## 📦 Bộ dữ liệu

**SLAKE-vqa-english** ([HuggingFace](https://huggingface.co/datasets/mdwiratathya/SLAKE-vqa-english))

| Split | Số mẫu | CLOSED | OPEN |
|---|---|---|---|
| Train | 4.919 | 33,5% | 66,5% |
| Test | 1.061 | 355 | 706 |

Bao phủ: X-quang, MRI, CT scan, siêu âm — não, ngực, bụng, cột sống.

---

## 🛠️ Công nghệ sử dụng

```
Python 3.10 · PyTorch 2.x · HuggingFace Transformers
open_clip_torch · peft · bitsandbytes · accelerate
datasets · rouge-score · nltk · sentence-transformers
Gradio (giao diện demo)
```

---

## 🚀 Hướng dẫn chạy

### 1. Clone & Cài đặt

```bash
git clone https://github.com/heavenjackson2k4/medical-vqa-vlm.git
cd medical-vqa-vlm
pip install -r requirements.txt
```

### 2. Tải dữ liệu

```python
from datasets import load_dataset
ds = load_dataset("mdwiratathya/SLAKE-vqa-english")
```

### 3. Huấn luyện — BiomedCLIP + Qwen2.5

```bash
jupyter notebook biomedclip-qwen-slake.ipynb
```

### 4. Huấn luyện — PaliGemma

```bash
jupyter notebook paligemma-medicalVQA_v2.ipynb
```

### 5. Chạy Demo

```bash
jupyter notebook demo-biomedclip-qwen.ipynb
# hoặc
jupyter notebook paligemma-demo.ipynb
```

---

## 📁 Cấu trúc thư mục

```
medical-vqa-vlm/
├── biomedclip-qwen-slake.ipynb      # Training: BiomedCLIP + Qwen2.5
├── demo-biomedclip-qwen.ipynb       # Inference demo: BiomedCLIP + Qwen2.5
├── paligemma-medicalVQA.ipynb       # Training: PaliGemma (v1)
├── paligemma-medicalVQA_v2.ipynb    # Training: PaliGemma (v2, cải tiến)
├── paligemma-demo.ipynb             # Inference demo: PaliGemma
├── requirements.txt
└── README.md
```

---

## 🧪 Các độ đo đánh giá

| Độ đo | Ý nghĩa |
|---|---|
| **Exact Match (EM)** | Khớp hoàn toàn sau chuẩn hóa (lowercase, bỏ ký tự đặc biệt) |
| **Token F1** | F1 theo token giữa dự đoán và ground truth |
| **BLEU-1 / BLEU-4** | Độ chồng lấp unigram / 4-gram |
| **ROUGE-L** | Độ tương đồng theo chuỗi con chung dài nhất (LCS) |
| **Cosine Similarity** | Tương đồng ngữ nghĩa qua embedding `all-MiniLM-L6-v2` |

Kết quả được báo cáo riêng cho câu hỏi **CLOSED** (Yes/No) và **OPEN** (câu trả lời tự do).

---

## 💡 Nhận xét chính

1. **Encoder chuyên biệt y tế quan trọng hơn kích thước LLM**: BiomedCLIP (pretrain 15M+ cặp PubMed) giúp mô hình hiểu đặc trưng ảnh y tế sâu hơn SigLIP tổng quát, kể cả sau fine-tuning.

2. **LLM nhỏ + encoder tốt > LLM lớn + encoder tổng quát**: Qwen2.5-0.5B + BiomedCLIP đánh bại PaliGemma 3B trên toàn bộ các độ đo trong cùng điều kiện phần cứng.

3. **CLOSED vs OPEN**: Cả hai mô hình xử lý tốt câu hỏi Yes/No (EM > 85%), nhưng câu hỏi OPEN — đặc biệt mô tả giải phẫu hoặc bệnh lý — vẫn còn nhiều không gian cải thiện.

4. **Label masking là bắt buộc**: Gán `-100` cho prompt tokens giúp mô hình học sinh câu trả lời thay vì lặp lại câu hỏi (echo effect).

---

## ⚠️ Hạn chế

- Dữ liệu huấn luyện nhỏ (~4.900 mẫu) — chưa đủ đa dạng để tổng quát hóa tốt
- Bị giới hạn bởi GPU T4 16GB (Kaggle free) — PaliGemma chỉ huấn luyện được 2 epoch
- Chưa đánh giá cross-dataset (VQA-RAD, PathVQA)

---

## 🔮 Hướng phát triển

- [ ] Tăng số epoch cho PaliGemma lên 10–15 epoch với phần cứng tốt hơn
- [ ] Tích hợp thêm VQA-RAD và PathVQA vào pipeline huấn luyện
- [ ] Tích hợp đồ thị tri thức y khoa vào bộ mã hóa câu hỏi
- [ ] Xây dựng giao diện demo Gradio/Streamlit
- [ ] Mở rộng sang Medical VQA tiếng Việt
- [ ] Đánh giá lâm sàng có sự tham gia của chuyên gia y tế

---

## 📚 Tài liệu tham khảo

1. Radford et al. (2021). [CLIP](https://arxiv.org/abs/2103.00020) — ICML 2021
2. Zhang et al. (2023). [BiomedCLIP](https://arxiv.org/abs/2303.00915) — Microsoft Research
3. Beyer et al. (2024). [PaliGemma](https://arxiv.org/abs/2407.07726) — Google DeepMind
4. Hu et al. (2022). [LoRA](https://arxiv.org/abs/2106.09685) — ICLR 2022
5. Li et al. (2023). [LLaVA-Med](https://arxiv.org/abs/2306.00890) — NeurIPS 2023
6. Liu et al. (2021). [SLAKE](https://arxiv.org/abs/2102.09542) — ISBI 2021
7. Bai et al. (2023). [Qwen](https://arxiv.org/abs/2309.16609) — Alibaba Cloud

---

## 👤 Tác giả

**Võ Trường Thiên** — Khoa học Máy tính, IUH  
📧 truongthienvo20042402@gmail.com  

---

<p align="center">
  <i>Đồ án cuối kỳ · Môn Xử Lý Ngôn Ngữ Tự Nhiên · Học kỳ II · 2025–2026</i>
</p>
