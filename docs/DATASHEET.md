# Datasheet for Dataset: PlantVillage (Foliar Disease Subset)

**Standard**: Gebru et al. (2021) *Datasheets for Datasets*, Communications of the ACM  
**Project**: Automated Crop Leaf Disease Identification and Precision Dosage Recommendation Pipeline  
**Dataset Name**: PlantVillage (Color Leaf Disease Dataset)  
**Curators**: David P. Hughes and Marcel Salathé (Penn State University & EPFL)  
**Audit & Splitting Date**: October 2026

---

## 1. Motivation

### For what purpose was the dataset created?
The PlantVillage dataset was created in 2015–2016 as an open-access image repository to facilitate computer vision research in automated plant pathology diagnosis, specifically to empower smallholder farmers via mobile smartphone screening.

### Who created the dataset and on behalf of which entity?
Created by Dr. David P. Hughes (Department of Entomology and Biology, Penn State University) and Dr. Marcel Salathé (School of Life Sciences and School of Computer and Communication Sciences, EPFL).

### Who funded the creation of the dataset?
Supported in part by Pennsylvania State University, EPFL, and the open-access PlantVillage initiative.

---

## 2. Composition

### What do the instances that comprise the dataset represent?
Each instance is a standardized digital photograph ($256 \times 256$ pixels, RGB) of a single detached crop leaf placed against a uniform sheet of paper (typically white, grey, or light blue) under diffuse indoor or greenhouse lighting.

### How many instances are there in total?
- **Raw Image Count**: 54,305 image files.
- **Deduplicated Image Count**: 54,284 images (21 exact SHA-256 duplicate copies removed during audit).
- **Partitioned Splits**:
  - **Train**: 38,002 images (70%)
  - **Validation**: 8,148 images (15%)
  - **Test**: 8,134 images (15%)

### What are the classes and distributions?
38 crop-condition classes across 14 crop species:
- **Crops**: Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.
- **Class Imbalance**: High imbalance with a **36:1 frequency ratio**.
  - Largest class: `Orange___Haunglongbing_(Citrus_greening)` ($N = 5,507$).
  - Smallest class: `Potato___healthy` ($N = 152$).

### What are the known structural gaps in composition?
1. **Disease-Only Gaps**: Orange (Citrus greening only) and Squash (Powdery mildew only) have **zero healthy reference samples**.
2. **Healthy-Only Gaps**: Blueberry, Raspberry, and Soybean contain **only healthy images** with zero disease conditions.
3. Together, single-condition crops constitute **26.0% of total dataset volume**.

---

## 3. Collection Process

### How was the data collected?
Leaves exhibiting visible symptoms of fungal, bacterial, or viral disease (as well as healthy control leaves) were physically excised from host plants in experimental agricultural research stations and commercial farms in Pennsylvania, USA, and partner facilities. Excised leaves were brought indoors, laid flat upon laboratory paper backdrops, and photographed using digital cameras under controlled ambient lighting.

### Who was involved in data collection and what were their roles?
Trained agricultural technicians and plant pathologists harvested and staged the leaves; photographers executed multi-shot capture sequences.

---

## 4. Preprocessing, Cleaning & Group Labeling

### What preprocessing or cleaning was performed?
1. **Deduplication**: SHA-256 hash calculation identified and expunged 21 identical duplicate files.
2. **Capture Group Token Extraction**: Filename regex patterns were extracted (e.g. `GH_HL Leaf 173.1`, `173.2`). Multiple images originating from the same leaf or shooting sequence share capture tokens.
3. **Crop-Prefixed Group Scoping**: Photographers reused generic sequence numbers across different crops (e.g. `RS_HL 5672` in Apple and Soybean). Groups were prefixed as `<crop>::<id>` to isolate distinct capture events.
4. **Group-Aware Stratified Splitting**: Partitioned train, validation, and test splits ensuring zero group intersection between train and test:
   $$\text{Train Groups} \cap \text{Test Groups} = \emptyset$$

---

## 5. Uses

### Has the dataset been used for previous tasks?
Extensively used in academic benchmarking (>3,000 citations). Widely utilized for evaluating convolutional neural networks (ResNet, MobileNet, EfficientNet) and Vision Transformers in agricultural leaf classification.

### What are known limitations or potential biases?
1. **Capture Paper Bias (Noyan et al., 2022)**: The background paper and lighting conditions correlate with disease classes. A baseline classifier trained solely on 4 corner background pixels achieved 46.1% accuracy, confirming spurious background memorization.
2. **Absence of In-Situ Field Context**: Images lack natural field complexity: no soil textures, direct sunlight glare, shadowing, insect chew holes, or overlapping adjacent foliage.
3. **Detached Leaf Morphology**: Excised leaves wilt rapidly, altering foliar turgor and reflectance characteristics compared to live field canopies.

---

## 6. Distribution

### How is the dataset distributed?
Freely available under public open-access academic repositories (e.g., Mendeley Data, Kaggle).

### What is the dataset license?
Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).

---

## 7. Maintenance

### Who maintains the dataset?
Maintained by the open-source agricultural research community. In this project repository, the partitioned manifest files (`data/splits/train.csv`, `val.csv`, `test.csv`) and SHA-256 checksums are version-controlled via DVC and Git.
