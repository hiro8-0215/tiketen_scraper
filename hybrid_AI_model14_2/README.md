# Model 14-2 — 学習量を削減しない効率最適化版

Model 14と同じQwen 7B、9,576 sold行、5-fold、5 epoch、最大384 token、
実効batch 32を維持します。精度条件を軽くせず、GPUへの流し方を最適化します。

## 最適化点

- micro-batch `2 -> 4`、gradient accumulation `16 -> 8`（実効batchは32のまま）
- PyTorch SDPA attention
- BF16 Tensor Core + TF32許可
- fused AdamW
- 非reentrant gradient checkpointing
- 長さ別バッチでpadding計算を削減
- 8の倍数paddingでTensor Core効率を改善
- 評価batch `4 -> 8`（評価件数・頻度は同一）
- DataLoader worker、pin memory、persistent worker、prefetch
- 評価epoch中はlossだけを収集し、OOF推論時だけlogitsを保存
- Model 14の検証済みfold/BERT成果物をハードリンク再利用
- `device_map={"": 0}`で全パラメータを専用GPUへ固定（CPU/disk offload禁止）
- CUDA allocatorを専用VRAMの90%に制限し、WDDM共有メモリへの退避を抑止
- 非CUDAパラメータを検出したら低速学習を開始せずエラーにする
- CUDA断片化対策とCPUスレッド上限で、RAM・CPUの競合を防止

現在走っているModel 14には影響しません。GPUを競合させないため、Model 14完了後に実行してください。

```powershell
python run_all.py
```

またはfold単位で実行できます。

```powershell
python bootstrap_artifacts.py
python 2_train_qwen_oof.py --fold 0
python 2_train_qwen_oof.py --fold 1
python 2_train_qwen_oof.py --fold 2
python 2_train_qwen_oof.py --fold 3
python 2_train_qwen_oof.py --fold 4
python 3_train_meta.py
```

専用VRAMの90%上限内でbatch 4が入らない場合は、自動的にbatch 2・
gradient accumulation 16へ再試行します。実効batch 32と学習量は変わらず、
CPU/shared-memory offloadは行いません。

実行中の専用・共有GPUメモリは別ターミナルから確認できます。

```powershell
powershell -ExecutionPolicy Bypass -File .\resource_status.ps1
```
