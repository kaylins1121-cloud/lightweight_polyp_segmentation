# Lightweight Polyp Segmentation

Framework huấn luyện end-to-end cho bài toán phân đoạn polyp đại tràng, thiết kế theo dạng **registry** cho model và loss để dễ mở rộng/cải tiến.

## Cấu trúc project

```
polyp-segmentation/
├── main.py                 # entrypoint CLI, huấn luyện end-to-end
├── config.py                # định nghĩa toàn bộ tham số CLI
├── test_inference.py        # inference + GradCAM từ checkpoint đã train
├── plot.py                   # vẽ biểu đồ loss/metric từ log CSV
├── requirements.txt
├── models/
│   ├── __init__.py           # auto-detect + registry (register_model)
│   └── unet.py                # baseline UNet (đăng ký tên "unet")
├── losses/
│   ├── __init__.py           # auto-detect + registry (register_loss)
│   └── dice_loss.py           # DiceLoss ("dice"), DiceBCELoss ("dice_bce")
└── utils/
    ├── seed.py                # seed & deterministic reproducibility
    ├── dataset.py              # PolypTrainDataset / PolypTestDataset
    ├── metrics.py               # mDice, mIoU, weighted F-measure, S-measure, mean/max E-measure, MAE
    ├── engine.py                 # train/eval loop, multi-scale training
    ├── gradcam.py                 # GradCAM
    ├── logger.py                  # AverageMeter, CSVLogger
    └── download_data.py            # tải dataset từ HuggingFace
```

## Thêm model / loss mới

Chỉ cần tạo file mới trong `models/` hoặc `losses/`, dùng decorator để đăng ký, không cần sửa bất kỳ đâu khác:

```python
# models/my_model.py
from models import register_model

@register_model("my_model")
class MyModel(nn.Module):
    ...
```

```python
# losses/my_loss.py
from losses import register_loss

@register_loss("my_loss")
class MyLoss(nn.Module):
    ...
```

`models/__init__.py` và `losses/__init__.py` tự động import mọi file `.py` trong thư mục để phát hiện các class đã đăng ký — `main.py --model my_model --loss my_loss` sẽ chạy được ngay.

## Cài đặt & chạy trên Google Colab

```python
!git clone <URL_REPO_CUA_BAN>
%cd <ten-repo>
!pip install -r requirements.txt

!python utils/download_data.py --out_dir dataset

!python main.py --model unet --loss dice
```

## Huấn luyện nhiều seed để đo độ ổn định (reproducibility)

```bash
!python main.py --model unet --loss dice --multiseed --seeds 42 123 456 789 2024
```

Kết quả từng seed + trung bình ± độ lệch chuẩn được lưu tại `logs/<model>_<loss>_multiseed_summary.json`.

## Tham số mặc định (theo bảng cấu hình thực nghiệm)

| Tham số | Giá trị |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Multi-scale training | [0.75, 1, 1.25] |
| Gradient clip | 0.5 |
| Decay rate | 0.1 (mỗi `--decay_epoch`, mặc định 50) |
| Weight decay | 1e-4 |
| Epochs | 100 |
| Input size | 352 × 352 |
| Batch size | 4 |

Toàn bộ có thể override qua CLI, ví dụ: `--lr 1e-3 --batch_size 8 --epochs 50`.

## Dữ liệu

Dataset: [Zyna1121/lightpranet-polyp-data](https://huggingface.co/datasets/Zyna1121/lightpranet-polyp-data) (`TrainDataset.zip`, `TestDataset.zip`), cấu trúc thư mục tham khảo theo [PraNet](https://github.com/DengPingFan/PraNet):

```
dataset/
├── TrainDataset/
│   ├── images/
│   └── masks/
└── TestDataset/
    ├── CVC-300/{images,masks}
    ├── CVC-ClinicDB/{images,masks}
    ├── CVC-ColonDB/{images,masks}
    ├── ETIS-LaribPolypDB/{images,masks}
    └── Kvasir/{images,masks}
```

Nếu cấu trúc thực tế sau khi giải nén khác đi (ví dụ không có thư mục con theo tên dataset), chỉnh lại `--test_path` trỏ thẳng tới thư mục chứa `images/` và `masks/`.

## Chỉ số đánh giá (tính trên mỗi ảnh test rồi lấy trung bình)

- **mDice** — Dice coefficient trung bình
- **mIoU** — Intersection-over-Union trung bình
- **wFmeasure (Fβω)** — weighted F-measure (Margolin et al., 2014)
- **Smeasure (Sα)** — structure measure (Fan et al., 2017)
- **meanEmeasure (mEξ)** / **maxEmeasure (maxEξ)** — enhanced-alignment measure, trung bình/lớn nhất theo 255 ngưỡng (Fan et al., 2018)
- **MAE** — mean absolute error

## GradCAM

Mặc định bật (`--gradcam`, tắt bằng `--no_gradcam`), tự lấy `model.gradcam_target_layer` (mỗi model tự khai báo layer mục tiêu) để sinh heatmap trên `--gradcam_samples` ảnh test, lưu tại `gradcam_outputs/<run_tag>/`.

## Reproducibility

`utils/seed.py` set seed cho `random`, `numpy`, `torch` (CPU + CUDA), bật `cudnn.deterministic`, tắt `cudnn.benchmark`, gọi `torch.use_deterministic_algorithms(True, warn_only=True)`, và dùng `worker_init_fn` + `torch.Generator` riêng cho `DataLoader` theo khuyến nghị của PyTorch. Lưu ý: xác định tuyệt đối 100% không được đảm bảo giữa các phiên bản PyTorch/phần cứng khác nhau, nhưng cùng môi trường + cùng seed sẽ cho kết quả gần như trùng khớp.

## Inference / GradCAM độc lập từ checkpoint

```bash
!python test_inference.py --checkpoint checkpoints/unet_dice_seed42_best.pth --model unet --gradcam
```

## Vẽ biểu đồ

```bash
!python plot.py --log_csv logs/unet_dice_seed42.csv
```

## License

MIT — xem [LICENSE](LICENSE).
