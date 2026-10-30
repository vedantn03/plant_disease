# Source Audit: Agricultural Advisory & Weather Rules Traceability Matrix

**Project**: Automated Crop Leaf Disease Identification and Precision Dosage Recommendation Pipeline
**Course**: B.Tech CSE-AI (MLOps Course Project, Syngenta Evaluation Pilot)
**Audit Version**: 2.0 (Phase 3 Expanded Audit)
**Policy**: No agronomic statement, chemical active ingredient, or weather trigger is asserted unless supported by a verified university cooperative extension, ICAR national institute, or government agricultural authority. Specific dosage rates are omitted unless published in the official cited guide; otherwise, the interface explicitly advises that dosage rates must be retrieved from the registered product label or local extension authority.

---

## 1. Verified Advisory Entries Matrix (Schema v2)

| Crop | Condition | Pathogen | Primary Extension Citation | Access Date | Section Heading & Supporting Text Extract | Regional Scope |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: |
| **Tomato** | `early_blight` (*Alternaria solani*) | Fungal | [UC IPM Tomato Pest Management Guidelines](https://ipm.ucanr.edu/PMG/r783300111.html) & [ICAR-IIHR Tomato Guidelines](https://iihr.res.in/) | 2026-10-14 | **Section: Early Blight Management**: *"Destroy or disk under crop debris promptly... Avoid overhead irrigation... Protectant fungicides such as chlorothalonil, mancozeb, and copper materials give good control if applied preventively."* | Global / India |
| **Tomato** | `late_blight` (*Phytophthora infestans*) | Fungal (Oomycete) | [Cornell CALS Vegetable Disease ID](https://plantdisease.net/tomato_late_blight) & [TNAU Agritech Portal](https://agritech.tnau.ac.in/) | 2026-10-14 | **Section: Late Blight on Tomato**: *"P. infestans can cause total crop loss in a few days under cool, wet conditions... Use certified disease-free transplants... Apply protectant fungicides (e.g. chlorothalonil, mancozeb) before disease onset."* | Global / India |
| **Tomato** | `bacterial_spot` (*Xanthomonas spp.*) | Bacterial | [UF/IFAS Extension Publication VH014](https://edis.ifas.ufl.edu/publication/VH014) | 2026-10-14 | **Section: Management of Bacterial Spot**: *"Use pathogen-free seed and transplants... Rotate crops with non-solanaceous hosts... Copper sprays mixed with mancozeb provide control where copper tolerance is moderate. Standard fungicides alone are ineffective."* | Global |
| **Potato** | `early_blight` (*Alternaria solani*) | Fungal | [UMaine Cooperative Extension Bulletin 2436](https://extension.umaine.edu/publications/2436e/) & [ICAR-CPRI Shimla](https://cpri.icar.gov.in/) | 2026-10-14 | **Section: Potato Early Blight**: *"Maintain optimum plant nitrogen and vigour... Begin fungicide applications when first lesions appear in the lower canopy... Rotate modes of action to delay resistance."* | Global / India |
| **Potato** | `late_blight` (*Phytophthora infestans*) | Fungal (Oomycete) | [UMaine Cooperative Extension Bulletin 2436](https://extension.umaine.edu/publications/2436e/) & [ICAR-CPRI Bulletin](https://cpri.icar.gov.in/) | 2026-10-14 | **Section: Potato Late Blight Control**: *"Destroy volunteer potato cull piles... A preventive spray schedule using protective fungicides is critical when weather forecasting models indicate high blight risk."* | Global / India |
| **Apple** | `apple_scab` (*Venturia inaequalis*) | Fungal | [Cornell CALS Tree Fruit Pathology](https://blogs.cornell.edu/plantpathhvl/apple-diseases/apple-scab/) & [Penn State Extension](https://extension.psu.edu/apple-scab) | 2026-10-14 | **Section: Apple Scab Primary Management**: *"Apply protectant fungicides (captan, mancozeb) starting at green tip. Rake or flail-mow overwintered leaves in autumn to reduce primary ascospore inoculum."* | Global / North America |
| **Grape** | `black_rot` (*Guignardia bidwellii*) | Fungal | [Ohio State University Extension FactSheet PLPATH-FRU-24](https://ohioline.osu.edu/factsheet/plpath-fru-24) & [ICAR-NRCG Pune](https://nrcgrapes.icar.gov.in/) | 2026-10-14 | **Section: Black Rot Management**: *"Remove mummified berries from the trellis during winter pruning. Early sprays from pre-bloom through post-bloom are critical to prevent fruit cluster infection."* | Global / India |
| **Corn (Maize)** | `common_rust_` (*Puccinia sorghi*) | Fungal | [Purdue University Extension BP-82-W](https://www.extension.purdue.edu/extmedia/BP/BP-82-W.pdf) | 2026-10-14 | **Section: Managing Corn Rust**: *"Plant resistant corn hybrids... Fungicide applications are rarely economical in field corn unless pustules appear on upper leaves before tasseling during cool, humid weather."* | Global / North America |

---

## 2. Weather Decision Rules Traceability Audit

All epidemiological risk triggers in `kb/weather_rules.yaml` map directly to published university and extension disease forecasting models:

| Rule ID | Crop / Condition | Environmental Threshold | Epidemiological Rationale & Cited Text Extract | Primary Authority |
| :--- | :--- | :--- | :--- | :--- |
| `tomato_late_blight_conducive` | **Tomato / Late Blight** | Temp: 15°C – 24°C<br>RH ≥ 80% | *"Moderate temperatures (15-24°C) combined with sustained relative humidity (>80%) provide optimal conditions for Phytophthora infestans sporangia germination and rapid secondary cycling."* | UC IPM Guidelines |
| `potato_late_blight_conducive` | **Potato / Late Blight** | Temp: 15°C – 22°C<br>RH ≥ 80% | *"Phytophthora infestans foliar infection occurs rapidly when leaf wetness persists for 8+ hours at 15-22°C."* (Wallin Blight Units criteria) | Cornell Cooperative Extension |
| `tomato_early_blight_warm_humid` | **Tomato / Early Blight** | Temp: 24°C – 30°C<br>RH ≥ 75% | *"Alternaria solani conidiophore sporulation is heavily favored by warm temperatures (24-30°C) and alternating wet/dry cycles."* (FAST model criteria) | UC IPM / Penn State |
| `apple_scab_mills_period` | **Apple / Scab** | Temp: 13°C – 24°C<br>RH ≥ 85% | *"Venturia inaequalis ascospore discharge and primary infection are favored by temperatures of 13-24°C under prolonged leaf wetness (Mills Period criteria)."* | Penn State Extension |
| `grape_black_rot_warm_wet` | **Grape / Black Rot** | Temp: 20°C – 28°C<br>RH ≥ 80% | *"Guignardia bidwellii ascospores require 6-7 hours of continuous leaf moisture at 20-28°C for successful tissue penetration."* | Cornell Fruit Resources |
| `corn_common_rust_cool_humid` | **Corn / Common Rust** | Temp: 16°C – 25°C<br>RH ≥ 85% | *"Puccinia sorghi urediniospores germinate optimally during cool, humid nights (16-25°C) with prolonged dew periods."* | Purdue Extension |

> [!IMPORTANT]
> **Dose Modification Prohibition**: Epidemiological weather rules serve exclusively to generate context-dependent alerts and escalate clinical urgency (`low` vs `elevated`). Weather rules NEVER dynamically scale, amplify, or modify chemical dosage concentrations.

---

## 3. Strict Non-Chemical and Regulatory Governance Principles

1. **Viral Pathogens (e.g. Tomato Mosaic Virus, Tomato Yellow Leaf Curl Virus)**:
   - **Negative Assertion**: FUNGICIDES MUST NEVER BE RECOMMENDED.
   - **Agronomic Grounding**: Plant viruses replicate intracellularly within the host symplast and are transmitted via mechanical vectors or insect vectors (e.g. *Bemisia tabaci* whiteflies). Fungicides have zero biochemical efficacy against viral coat proteins or RNA genomes; applying them causes financial waste, farmer exposure, and off-target ecotoxicity.
2. **Phloem-Limited Bacterial Infections (e.g. Citrus Greening / Huanglongbing)**:
   - **Negative Assertion**: NO CURATIVE CHEMICALS OR FUNGICIDES PERMITTED.
   - **Agronomic Grounding**: Caused by unculturable, fastidious phloem bacteria (*Candidatus Liberibacter* asiaticus) transmitted by psyllids (*Diaphorina citri*). No foliar fungicide or chemical spray is curative; management is strictly quarantined vector control and infected tree rogueing.
3. **Arthropod Pest Infestations (e.g. Two-Spotted Spider Mite)**:
   - **Negative Assertion**: NO FUNGICIDES PERMITTED.
   - **Agronomic Grounding**: *Tetranychus urticae* is an arachnid mite. Broad-spectrum fungicides can eradicate beneficial entomopathogenic fungi (e.g. *Neozygites floridana*) that naturally suppress mite populations, inducing secondary pest flare-ups.
4. **Dosage Information Policy**:
   - In strict compliance with regulatory governance, this pipeline does not fabricate active ingredient dosages. Users are directed to official EPA / Central Insecticides Board (CIB) registered product labels for mandatory volume, PPE, and Pre-Harvest Interval (PHI) directions.
