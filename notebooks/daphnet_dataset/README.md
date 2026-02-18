# Daphnet Dataset - FoG Detection Pipeline

Pipeline completo de análisis del dataset Daphnet para detección de Freezing of Gait (FoG).

---

## 📊 Pipeline Principal (Ejecutar en orden)

### 01_exploratory_data_analysis.ipynb 📈
**Análisis Exploratorio**
- Output: Insights sobre el dataset

### 02_preprocessing_and_windowing.ipynb 🔧  
**Ventanas Deslizantes**
- Output: `../../outputs/datasets_csv/daphnet_loso_windows_binary.pkl`

### 03_feature_extraction.ipynb ⚙️
**Extracción de Features**
- Features: Time + Frequency + Wavelet + Nonlinear (~197 features)
- Output: `../../outputs/daphnet_features/fold_subj_SXX/*.csv`

### 04_loso_pipeline_and_training.ipynb 🤖
**Entrenamiento LOSO**
- Pipeline: Imputer → Scaler → ADASYN → Random Forest
- Output: `../../outputs/daphnet_loso_results_binary.csv`

---

## 🚀 Ejecución Rápida

```bash
# 1. Ejecutar notebooks en orden: 01 → 02 → 03 → 04
# 2. Verificar outputs generados
ls -R ../../outputs/
```

**¡Listo para ejecutar! ✅**
