# GroundedSurg: A Multi-Procedure Benchmark for Language-Conditioned Surgical Tool Segmentation

Official repository for **GroundedSurg**, a grounding-based surgical vision benchmark introduced at MICCAI.

---

## 🧠 Overview

GroundedSurg reformulates surgical instrument perception as a **language-conditioned, instance-level segmentation task**.

Unlike conventional category-level segmentation benchmarks, GroundedSurg requires models to:

- Resolve natural-language references
- Disambiguate between multiple similar instruments
- Perform structured spatial grounding
- Produce precise pixel-level segmentation masks

---

## 🔥 Key Contributions

- First **language-conditioned surgical grounding benchmark**
- Instance-level disambiguation across multi-instrument scenes
- Structured spatial grounding:
  - Bounding box
  - Center point
  - Pixel-level mask
- Multi-procedure diversity
- Unified evaluation protocol for Vision-Language Models

---

## 📊 Dataset Statistics

| Statistic | Value |
|------------|--------|
| Number of images | ~612 |
| Tool annotations | ~1,071 |
| Average tools per image | ~1.6 |
| Surgical procedures | 4 |
| Annotation type | Pixel-level segmentation |
| Spatial grounding | Bounding box + Center point |
| Language descriptions | Instance-level |

---

## 🏥 Covered Procedures

- Ophthalmic Surgery  
- Laparoscopic Cholecystectomy  
- Robotic Nephrectomy  
- Gastrectomy  

---

## 🎯 Task Definition

Each benchmark instance consists of:

- Surgical image `I`
- Natural-language query `T`
- Bounding box `B`
- Center point `C`
- Ground-truth segmentation mask `M`

Objective:

```math
f(I, T, B, C) → \hat{M}
```

Models must localize and segment the instrument described by the query.

---

## 📈 Evaluation Metrics

### Region-Based Metrics
- IoU
- IoU@0.5 / IoU@0.9
- mIoU
- Dice

### Localization Metrics
- Bounding Box IoU
- Normalized Distance Error (NDE)

All metrics are computed per image-query pair.

---

## 🏗 Evaluation Pipeline

GroundedSurg follows a unified language-conditioned segmentation protocol:

1. Vision-Language Model predicts:
   - Bounding box
   - Center point
2. Predictions projected to segmentation backend (SAM2 / SAM3)
3. Final mask evaluated against ground truth

---

## 🤖 Evaluated Models

GroundedSurg benchmarked:

### Open-Source Models
- Qwen2.5-VL
- Qwen3-VL
- Gemma 3 (12B / 27B)
- LLaMA 3 Vision
- DeepSeek-VL2
- Mistral 3

### Reasoning-Oriented Models
- VisionReasoner
- Migician
- InternVL

### Medical-Domain Models
- MedMO
- MedGemma
- MedVLM-R1
- BiMediX2

### Closed-Source Models
- GPT-4o-mini
- GPT-5.2

---

## 📁 Repository Structure

```bash
GroundedSurg/
├── Model_Evaluation_Scripts/
│   ├── gemma3.py
│   ├── llama.py
│   ├── qwen_2_5.py
│   ├── qwen_3.py
│   ├── mistral_3.py
│   ├── intern_eval.py
│   ├── med_mo.py
│   ├── migician.py
│   ├── MedGemma/
│   ├── MedVLM-R1/
│   └── Segmentation/
│       ├── sam2.py
│       ├── sam3.py
│       └── mask_eval.sh
└── Prompt/
    ├── prompt1.txt
    └── prompt2.txt
```

---

## Bencharmark overview

![Qualitative Results](assets/benchmark_overview.png)
## 📸 Qualitative Results

![Qualitative Results](assets/qualitative.png)


---

## 🚀 Running Evaluation

Example:

```bash
python Model_Evaluation_Scripts/qwen_2_5.py
```

For segmentation backend:

```bash
python Model_Evaluation_Scripts/Segmentation/sam3.py
```

---

## 📦 Installation

```bash
conda create -n groundedsurg python=3.10
conda activate groundedsurg
pip install -r requirements.txt
```

---

## 📌 Citation

If you use GroundedSurg, please cite:

```bibtex
@inproceedings{groundedsurg2026,
  title={GroundedSurg: A Multi-Procedure Benchmark for Language-Conditioned Surgical Tool Segmentation},
  author={Tajamul Ashraf, Abrar ul Riyz, Wasif Tak , Tavaheed Tariq, Sonia Yadav, Moloud Abdar, Janibul Bashir},
  booktitle={MICCAI},
  year={2026}
}
```

---

## 📜 License

(To be added)

---

## ✉ Contact

For questions:
Abrar ul Riyaz
Gaash Research Lab  
NIT Srinagar  

---

## ⭐ If you find this work useful, please consider starring the repository.
