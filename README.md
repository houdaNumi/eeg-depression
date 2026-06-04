# EEG Depression Detection

## Description
Pipeline ML pour détecter la dépression à partir de signaux EEG frontaux.
Reproduction partielle du paper : "A machine learning based depression screening framework using temporal domain features of EEG signals" (2024).

## Dataset
MODMA — Multi-modal Open Dataset for Mental-disorder Analysis  
Lien : https://figshare.com/articles/dataset/EEG_Data_New/4244171  
55 participants (30 MDD, 25 sains) — fichiers EC (yeux fermés)

## Pipeline
1. Pre-processing — filtre notch 50Hz (MNE)
2. Feature extraction — 12 features temporelles par fenêtres de 10s
3. Feature selection — SelectKBest (6 features)
4. Normalisation — StandardScaler
5. Classification — SVM

## Résultats
| Modèle | Accuracy |
|--------|----------|
| KNN | 62.6% |
| KNN + feature selection | 66.4% |
| SVM + feature selection + normalisation | 70.0% |

## Limites
- Score du paper (96%) probablement surestimé — 55 sujets seulement
- Features temporelles uniquement — pas de features fréquentielles

## Technologies
Python 3.12 — MNE, NumPy, Scikit-learn, Pandas, SciPy