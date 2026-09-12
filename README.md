# Explainable AI for Diabetic Retinopathy Screening in Rural India (SIH 2026)

> **IMPORTANT CLINICAL & RESEARCH NOTICE**  
> - **Research Prototype:** This software is an academic research prototype developed for Smart India Hackathon (SIH) 2026.  
> - **Decision Support Only:** This AI system provides screening decision support, **not a clinical diagnosis**. Final clinical decisions must always be made by qualified ophthalmologists or medical professionals.  
> - **No Clinical Certification:** This system has not received regulatory medical device certification and makes no claims of clinical validation or superiority over commercial diagnostic systems.  
> - **Online Demonstration:** The public Streamlit Cloud deployment is an **online demonstration**. True offline edge deployment (for rural PHCs without internet) is an ongoing research track evaluated via local hardware simulations.  
> - **Data Privacy:** **No raw patient retinal images or training datasets are included** in this repository.

---

## 1. System Architecture

The screening pipeline processes retinal fundus images through four coordinated modules:

```
[Retinal Fundus Image]
         │
         ▼
[1. Quality Assessment Gate] ──(Inadequate)──► [Immediate Recapture Alert & Stop]
         │ (Acceptable)
         ▼
[2. EfficientNet-B0 Classifier] ─────────────► [5-Class DR Severity & Confidence]
         │
         ├──► [3. Grad-CAM Visual Explainability] (Heatmap attention overlay)
         │
         └──► [4. Clinical Referral Engine] ─────► [Referral Priority & Action Pathway]
```

### DR Severity Grading (ICDR Scale)
- **Class 0:** No DR
- **Class 1:** Mild NPDR (Non-Proliferative Diabetic Retinopathy)
- **Class 2:** Moderate NPDR
- **Class 3:** Severe NPDR
- **Class 4:** PDR (Proliferative Diabetic Retinopathy)

---

## 2. Model Performance Baseline

The protected baseline model checkpoint (`best_model_combined_v1.pth`, 15.60 MB) was evaluated on the frozen APTOS benchmark set:
- **Test Accuracy:** **67.87%**
- **Macro Precision:** **63.72%**
- **Macro Recall:** **61.61%**
- **Macro F1-Score:** **61.23%**

---

## 3. Local Installation & Execution

To run the application locally on your workstation:

```bash
# 1. Clone the repository
git clone https://github.com/haridharshan020-ux/CODE-HEIST.git
cd CODE-HEIST

# 2. Install dependencies (CPU PyTorch)
pip install -r requirements.txt

# 3. Launch the Streamlit application
streamlit run app.py
```

Open `http://localhost:8501` in any web browser. The interface is responsive across desktop, tablet, and mobile browsers, with built-in printable clinical report formatting.

---

## 4. Verification & Testing

Automated consistency and regression test suites are provided to verify system integrity:

```bash
# Run 10/10 regression consistency verification
python verify_consistency.py

# Run Streamlit AppTest UI test suite
python test_app_ui.py
```

---

## 5. Rural Deployment Feasibility Research (SIH Objective 10)

For rural primary health centres (PHCs) with intermittent connectivity and low-resource hardware, offline deployment feasibility was evaluated via Python-based simulation benchmarks. The benchmark scripts, raw measurement logs, and generated performance charts are preserved in the `deployment_benchmark/` directory.
