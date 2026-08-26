# Lightweight Polyp Segmentation

## Overview
Repository chuẩn nghiên cứu cho đề tài **Lightweight Polyp Segmentation**, hỗ trợ tự động tải dataset từ Hugging Face, quản lý model bằng cơ chế Plugin/Registry linh hoạt, train end-to-end trên GPU Google Colab (Tesla T4) và đánh giá trên 5 bộ test dataset.

## Dataset
Dataset được cấu hình tự động tải từ Hugging Face Hub:
* **Repository:** [Zyna1121/lightpranet-polyp-data](https://huggingface.co/datasets/Zyna1121/lightpranet-polyp-data)

## Installation & Google Colab Quickstart
Để chạy trên Google Colab với GPU Tesla T4, hãy thực hiện lần lượt các cell dưới đây.

### 1. Clone Repository & Cài đặt Thư viện
```bash
!git clone [https://github.com/doantrongthai/lightweight_polyp_segmentation.git](https://github.com/doantrongthai/lightweight_polyp_segmentation.git)
%cd lightweight_polyp_segmentation
!pip install -r requirements.txt
lightweight_polyp_segmentation/
├── main.py
├── requirements.txt
├── README.md
├── configs/
│   └── train.yaml
├── datasets/
│   └── polyp_dataset.py
├── models/
│   ├── __init__.py
│   └── unet.py
├── losses/
│   └── loss.py
├── metrics/
│   └── segmentation.py
└── utils/
    └── helpers.py
