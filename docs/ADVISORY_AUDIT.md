# Advisory provenance audit

This document records the provenance audit and current status of WP8 advisory content. A fetched page supports only the passage and location explicitly noted here. Entries whose source or passage could not be traced are placeholders in the active knowledge base until reviewed.

## Audit outcome

| Audit status | Count | Active knowledge-base status |
|---|---:|---|
| VERIFIED | 1 | One source-backed entry remains VERIFIED; reviewer is blank, so evidence grade B applies. |
| UNVERIFIED-SOURCE | 25 | Disease entries whose source or passage was not traceable; active KB status is PLACEHOLDER. |
| PLACEHOLDER | 12 | Healthy-class entries with no source claim; remain PLACEHOLDER. |

The active YAML contains 38 entries: 1 `VERIFIED` and 37 `PLACEHOLDER`. The verified common-rust entry has no reviewer and is grade B. All placeholders have empty scouting, cultural, and chemical guidance and direct users to consult a local agronomist.

The original `VERIFIED` labels were not treated as proof of source verification. Most pages were inaccessible, redirected to a different disease, or lacked a matching cited passage. `plantdisease.net` is not a Cornell domain; its Cornell attribution was removed. No guidance was reconstructed from memory.

## Fetched source pages

| Entry | URL | Publisher/page observed | Result |
|---|---|---|---|
| Tomato early blight | https://ipm.ucanr.edu/PMG/r783300111.html | UC IPM page redirects to Tomato Fruitworm | Mismatch; cannot support early-blight advice. |
| Tomato late blight | https://plantdisease.net/tomato_late_blight | Not fetchable | Unverified; not a Cornell URL. |
| Tomato bacterial spot | https://edis.ifas.ufl.edu/publication/VH014 | Not fetchable | Unverified. |
| Potato early blight | https://extension.umaine.edu/publications/2436e/ | University of Maine, “Potato Facts: Powdery Scab of Potatoes” | Mismatch; cannot support early-blight advice. |
| Potato late blight | https://ag.umass.edu/vegetable/fact-sheets/potato-late-blight | Not fetchable | Unverified; check this URL manually. |
| Corn common rust | https://www.extension.purdue.edu/extmedia/BP/BP-82-W.pdf | Purdue Extension, “Common and Southern Rusts” | Page 4, “Managing the Disease”: hybrid resistance ratings and early planting can reduce risk; fungicides are available for in-season rust management; follow label directions/restrictions. The old detailed chemical/rate/weather guidance is not supported by this page and must not be retained. |
| Grape Esca | https://ipm.ucanr.edu/PMG/r302100511.html | UC IPM Esca page | Page text describes the disease complex and notes that wood infection cannot be eradicated by fungicides; existing timing and chemical advice is not supported. |
| Tomato spider mites | https://ipm.ucanr.edu/PMG/r783400111.html | UC IPM page redirects to Tomato Russet Mite | Mismatch; cannot support two-spotted spider-mite advice. |
| Remaining cited disease URLs | See the source URL in the entry-by-entry inventory below | Several returned access errors or were unavailable | Unverified-source; do not use as guidance until manually checked. |

## Entry-by-entry inventory

The table below was reconciled against every record in `kb/advisories.yaml`; each crop/condition pair appears once. `UNVERIFIED-SOURCE` is the provenance finding, while the active KB status is `PLACEHOLDER`. Healthy-class placeholders make no source or disease-guidance claims.

| Crop | Condition | Audit status | Active KB status | Source URL (audited) | Publisher recorded in original KB | Guidance scripted/template-generated | Traceable to cited page |
|---|---|---|---|---|---|---|---|
| tomato | early_blight | UNVERIFIED-SOURCE | PLACEHOLDER | https://ipm.ucanr.edu/PMG/r783300111.html | UC IPM | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| tomato | late_blight | UNVERIFIED-SOURCE | PLACEHOLDER | https://plantdisease.net/tomato_late_blight | Cornell Cooperative Extension (incorrect attribution) | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| tomato | bacterial_spot | UNVERIFIED-SOURCE | PLACEHOLDER | https://edis.ifas.ufl.edu/publication/VH014 | University of Florida IFAS | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| potato | early_blight | UNVERIFIED-SOURCE | PLACEHOLDER | https://extension.umaine.edu/publications/2436e/ | University of Maine Cooperative Extension | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| potato | late_blight | UNVERIFIED-SOURCE | PLACEHOLDER | https://ag.umass.edu/vegetable/fact-sheets/potato-late-blight | University of Massachusetts Amherst Extension | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| apple | healthy | PLACEHOLDER | PLACEHOLDER | — | — | Yes; legacy generator scripts | Not applicable; no source claimed. |
| blueberry | healthy | PLACEHOLDER | PLACEHOLDER | — | — | Yes; legacy generator scripts | Not applicable; no source claimed. |
| cherry_(including_sour) | healthy | PLACEHOLDER | PLACEHOLDER | — | — | Yes; legacy generator scripts | Not applicable; no source claimed. |
| corn_(maize) | healthy | PLACEHOLDER | PLACEHOLDER | — | — | Yes; legacy generator scripts | Not applicable; no source claimed. |
| grape | healthy | PLACEHOLDER | PLACEHOLDER | — | — | Yes; legacy generator scripts | Not applicable; no source claimed. |
| peach | healthy | PLACEHOLDER | PLACEHOLDER | — | — | Yes; legacy generator scripts | Not applicable; no source claimed. |
| pepper,_bell | healthy | PLACEHOLDER | PLACEHOLDER | — | — | Yes; legacy generator scripts | Not applicable; no source claimed. |
| potato | healthy | PLACEHOLDER | PLACEHOLDER | — | — | Yes; legacy generator scripts | Not applicable; no source claimed. |
| raspberry | healthy | PLACEHOLDER | PLACEHOLDER | — | — | Yes; legacy generator scripts | Not applicable; no source claimed. |
| soybean | healthy | PLACEHOLDER | PLACEHOLDER | — | — | Yes; legacy generator scripts | Not applicable; no source claimed. |
| strawberry | healthy | PLACEHOLDER | PLACEHOLDER | — | — | Yes; legacy generator scripts | Not applicable; no source claimed. |
| tomato | healthy | PLACEHOLDER | PLACEHOLDER | — | — | Yes; legacy generator scripts | Not applicable; no source claimed. |
| apple | apple_scab | UNVERIFIED-SOURCE | PLACEHOLDER | https://blogs.cornell.edu/plantpathhvl/apple-diseases/apple-scab/ | Cornell University | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| apple | black_rot | UNVERIFIED-SOURCE | PLACEHOLDER | https://blogs.cornell.edu/plantpathhvl/apple-diseases/black-rot/ | Cornell University | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| apple | cedar_apple_rust | UNVERIFIED-SOURCE | PLACEHOLDER | https://extension.psu.edu/cedar-apple-rust | Penn State Extension | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| cherry_(including_sour) | powdery_mildew | UNVERIFIED-SOURCE | PLACEHOLDER | https://treefruit.wsu.edu/crop-protection/opm/cherry-powdery-mildew/ | Washington State University Extension | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| corn_(maize) | cercospora_leaf_spot gray_leaf_spot | UNVERIFIED-SOURCE | PLACEHOLDER | https://www.extension.purdue.edu/extmedia/BP/BP-56-W.pdf | Purdue Extension | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| corn_(maize) | common_rust_ | VERIFIED | VERIFIED | https://www.extension.purdue.edu/extmedia/BP/BP-82-W.pdf | Purdue Extension | Yes; legacy generator scripts | Yes — Purdue Extension BP-82-W, page 4, Managing the Disease section. |
| corn_(maize) | northern_leaf_blight | UNVERIFIED-SOURCE | PLACEHOLDER | https://extension.entm.purdue.edu/fieldcropsipm/diseases/northern-corn-leaf-blight.php | Purdue Extension | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| grape | black_rot | UNVERIFIED-SOURCE | PLACEHOLDER | https://ohioline.osu.edu/factsheet/plpath-fru-24 | Ohio State University Extension | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| grape | esca_(black_measles) | UNVERIFIED-SOURCE | PLACEHOLDER | https://ipm.ucanr.edu/PMG/r302100511.html | UC IPM | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| grape | leaf_blight_(isariopsis_leaf_spot) | UNVERIFIED-SOURCE | PLACEHOLDER | https://nrcgrapes.icar.gov.in/ | ICAR NRCG | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| orange | haunglongbing_(citrus_greening) | UNVERIFIED-SOURCE | PLACEHOLDER | https://crec.ifas.ufl.edu/research/citrus-production/disease-management/huanglongbing/ | University of Florida IFAS | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| peach | bacterial_spot | UNVERIFIED-SOURCE | PLACEHOLDER | https://extension.psu.edu/bacterial-spot-of-peach | Penn State Extension | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| pepper,_bell | bacterial_spot | UNVERIFIED-SOURCE | PLACEHOLDER | https://agritech.tnau.ac.in/crop_protection/crop_prot_crop_diseases_veg_chillies.html | Tamil Nadu Agricultural University | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| squash | powdery_mildew | UNVERIFIED-SOURCE | PLACEHOLDER | https://vegetablemdonline.ppath.cornell.edu/factsheets/Cucurbits_PM.htm | Cornell University | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| strawberry | leaf_scorch | UNVERIFIED-SOURCE | PLACEHOLDER | https://extension.psu.edu/strawberry-diseases-leaf-scorch | Penn State Extension | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| tomato | leaf_mold | UNVERIFIED-SOURCE | PLACEHOLDER | https://plantdisease.net/tomato_leaf_mold | Publisher claim not independently verified | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| tomato | septoria_leaf_spot | UNVERIFIED-SOURCE | PLACEHOLDER | https://www.extension.purdue.edu/extmedia/BP/BP-22-W.pdf | Purdue Extension | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| tomato | spider_mites two-spotted_spider_mite | UNVERIFIED-SOURCE | PLACEHOLDER | https://ipm.ucanr.edu/PMG/r783400111.html | UC IPM | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| tomato | target_spot | UNVERIFIED-SOURCE | PLACEHOLDER | https://edis.ifas.ufl.edu/publication/PP316 | University of Florida IFAS | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| tomato | tomato_yellow_leaf_curl_virus | UNVERIFIED-SOURCE | PLACEHOLDER | https://ipm.ucanr.edu/PMG/r783101611.html | UC IPM | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |
| tomato | tomato_mosaic_virus | UNVERIFIED-SOURCE | PLACEHOLDER | https://vegetablemdonline.ppath.cornell.edu/factsheets/Virus_Tomato.htm | Cornell CALS | Yes; legacy generator scripts | No; cited source or matching passage was not verified in this audit. |

## Manual source checks requested

The source URLs still requiring manual checks are listed in the entry table above. For convenience, these 25 unverified disease-entry URLs are repeated here:

<!-- UNVERIFIED_URLS_BEGIN -->
- tomato / `early_blight`: https://ipm.ucanr.edu/PMG/r783300111.html
- tomato / `late_blight`: https://plantdisease.net/tomato_late_blight
- tomato / `bacterial_spot`: https://edis.ifas.ufl.edu/publication/VH014
- potato / `early_blight`: https://extension.umaine.edu/publications/2436e/
- potato / `late_blight`: https://ag.umass.edu/vegetable/fact-sheets/potato-late-blight
- apple / `apple_scab`: https://blogs.cornell.edu/plantpathhvl/apple-diseases/apple-scab/
- apple / `black_rot`: https://blogs.cornell.edu/plantpathhvl/apple-diseases/black-rot/
- apple / `cedar_apple_rust`: https://extension.psu.edu/cedar-apple-rust
- cherry_(including_sour) / `powdery_mildew`: https://treefruit.wsu.edu/crop-protection/opm/cherry-powdery-mildew/
- corn_(maize) / `cercospora_leaf_spot gray_leaf_spot`: https://www.extension.purdue.edu/extmedia/BP/BP-56-W.pdf
- corn_(maize) / `northern_leaf_blight`: https://extension.entm.purdue.edu/fieldcropsipm/diseases/northern-corn-leaf-blight.php
- grape / `black_rot`: https://ohioline.osu.edu/factsheet/plpath-fru-24
- grape / `esca_(black_measles)`: https://ipm.ucanr.edu/PMG/r302100511.html
- grape / `leaf_blight_(isariopsis_leaf_spot)`: https://nrcgrapes.icar.gov.in/
- orange / `haunglongbing_(citrus_greening)`: https://crec.ifas.ufl.edu/research/citrus-production/disease-management/huanglongbing/
- peach / `bacterial_spot`: https://extension.psu.edu/bacterial-spot-of-peach
- pepper,_bell / `bacterial_spot`: https://agritech.tnau.ac.in/crop_protection/crop_prot_crop_diseases_veg_chillies.html
- squash / `powdery_mildew`: https://vegetablemdonline.ppath.cornell.edu/factsheets/Cucurbits_PM.htm
- strawberry / `leaf_scorch`: https://extension.psu.edu/strawberry-diseases-leaf-scorch
- tomato / `leaf_mold`: https://plantdisease.net/tomato_leaf_mold
- tomato / `septoria_leaf_spot`: https://www.extension.purdue.edu/extmedia/BP/BP-22-W.pdf
- tomato / `spider_mites two-spotted_spider_mite`: https://ipm.ucanr.edu/PMG/r783400111.html
- tomato / `target_spot`: https://edis.ifas.ufl.edu/publication/PP316
- tomato / `tomato_yellow_leaf_curl_virus`: https://ipm.ucanr.edu/PMG/r783101611.html
- tomato / `tomato_mosaic_virus`: https://vegetablemdonline.ppath.cornell.edu/factsheets/Virus_Tomato.htm
<!-- UNVERIFIED_URLS_END -->

Before any entry is marked VERIFIED, manually open its source URL and compare each retained sentence against the cited page. Start with the entries that will be shown in the demo: tomato early blight, tomato late blight, potato late blight, tomato bacterial spot, and corn common rust. For each, record the exact page title, publisher, passage heading/page, access date, and reviewer in `kb/advisories.yaml`. Do not restore removed dosages or product names unless the source passage directly contains them and local registration review is completed.

Weather rules currently have no cited source URLs. Until sources are added and verified, the API must report weather risk as unknown and must not apply the numeric rules.
