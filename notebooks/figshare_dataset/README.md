# Figshare Dataset - Freeze Detection Pipeline

Pipeline completo de análisis del dataset Figshare para detección de Freeze episodes.

---

## 📊 Pipeline Principal (Ejecutar en orden)

### 01_exploratory_data_analysis.ipynb 📈
**Análisis Exploratorio**
- Output: Insights sobre el dataset

### 02_preprocessing_and_windowing.ipynb 🔧  
**Ventanas Deslizantes**
- Output: `../../outputs/datasets_csv/figshare_loso_windows_binary.pkl`

### 03_feature_extraction.ipynb ⚙️
**Extracción de Features**
- Features: Time + Frequency + Wavelet + Nonlinear (~197 features)
- Output: `../../outputs/figshare_features/fold_subj_SXX/*.csv`

### 04_loso_pipeline_and_training.ipynb 🤖
**Entrenamiento LOSO**
- Pipeline: Imputer → Scaler → ADASYN → Random Forest
- Output: `../../outputs/figshare_loso_results_binary.csv`

---

## 🚀 Ejecución Rápida

```bash
# 1. Ejecutar notebooks en orden: 01 → 02 → 03 → 04
# 2. Verificar outputs generados
ls -R ../../outputs/
```

**¡Listo para ejecutar! ✅**
