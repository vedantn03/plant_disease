# Advisory provenance audit

This is a read-only audit of the advisory content present at the start of WP8. “Fetched” means the cited page was opened during this audit; it does not certify guidance beyond the passage named below. Entries whose cited text could not be checked are not safe for advice and must be downgraded to placeholders until reviewed.

## Audit outcome

| Original status | Count | WP8 disposition |
|---|---:|---|
| VERIFIED in the original YAML | 26 | Only source text directly matched to a fetched page may retain VERIFIED; other entries become PLACEHOLDER. |
| PLACEHOLDER in the original YAML | 12 | Remain placeholders. |

After the WP8 content migration, the active YAML contains 1 `VERIFIED` entry and 37 `PLACEHOLDER` entries. The retained common-rust entry has reviewer blank and therefore evidence grade B. The other 37 entries have no guidance text, chemical options, or source URL and direct the user to consult a local agronomist.

The original `VERIFIED` labels are not evidence of source verification. Most pages were inaccessible, redirected to a different disease, or were not traceable to the claimed publisher. `plantdisease.net` is not a Cornell domain; the attribution “Cornell Cooperative Extension” is removed from the record. No advice below is reconstructed from memory.

## Fetched source pages

| Entry | URL | Publisher/page observed | Result |
|---|---|---|---|
| Tomato early blight | https://ipm.ucanr.edu/PMG/r783300111.html | UC IPM page redirects to Tomato Fruitworm | Mismatch; cannot support early-blight advice. |
| Tomato late blight | https://plantdisease.net/tomato_late_blight | Not fetchable | Unverified; not a Cornell URL. |
| Tomato bacterial spot | https://edis.ifas.ufl.edu/publication/VH014 | Not fetchable | Unverified. |
| Potato early blight | https://extension.umaine.edu/publications/2436e/ | University of Maine, “Potato Facts: Powdery Scab of Potatoes” | Mismatch; cannot support early-blight advice. |
| Potato late blight | https://extension.umaine.edu/publications/2436e/ | University of Maine, “Potato Facts: Powdery Scab of Potatoes” | Mismatch; cannot support late-blight advice. |
| Corn common rust | https://www.extension.purdue.edu/extmedia/BP/BP-82-W.pdf | Purdue Extension, “Common and Southern Rusts” | Page 4, “Managing the Disease”: hybrid resistance ratings and early planting can reduce risk; fungicides are available for in-season rust management; follow label directions/restrictions. The old detailed chemical/rate/weather guidance is not supported by this page and must not be retained. |
| Grape Esca | https://ipm.ucanr.edu/PMG/r302100511.html | UC IPM Esca page | Page text describes the disease complex and notes that wood infection cannot be eradicated by fungicides; existing timing and chemical advice is not supported. |
| Tomato spider mites | https://ipm.ucanr.edu/PMG/r783400111.html | UC IPM page redirects to Tomato Russet Mite | Mismatch; cannot support two-spotted spider-mite advice. |
| Remaining cited disease URLs | See the source URL in the corresponding row below | Several returned access errors or were unavailable | Unverified-source; do not use as guidance until manually checked. |

## Entry-by-entry inventory

The generator scripts contain programmatic/static templates. The record texts do not include source passage locations, so unless the fetched match is explicitly identified above, the text is not traceable to a page. “Generated/template” indicates the entry is assembled or rewritten in a script; it does not mean source text was verified.

| Crop / condition | Original status | Source URL | Publisher as recorded | Script-generated/template | Traceable passage |
|---|---|---|---|---|---|
| tomato / early_blight | VERIFIED | https://ipm.ucanr.edu/PMG/r783300111.html | UC IPM | Yes | No; redirects to fruitworm |
| tomato / late_blight | VERIFIED | https://plantdisease.net/tomato_late_blight | Cornell Cooperative Extension (incorrect attribution) | Yes | No; URL not Cornell |
| tomato / bacterial_spot | VERIFIED | https://edis.ifas.ufl.edu/publication/VH014 | University of Florida IFAS | Yes | No; page unavailable |
| potato / early_blight | VERIFIED | https://extension.umaine.edu/publications/2436e/ | University of Maine Cooperative Extension | Yes | No; page is about powdery scab |
| potato / late_blight | VERIFIED | https://ag.umass.edu/vegetable/fact-sheets/potato-late-blight | University of Massachusetts Amherst Extension | Yes | No; page unavailable |
| apple / healthy | PLACEHOLDER | — | — | Yes | Not applicable |
| blueberry / healthy | PLACEHOLDER | — | — | Yes | Not applicable |
| cherry_(including_sour) / healthy | PLACEHOLDER | — | — | Yes | Not applicable |
| corn_(maize) / healthy | PLACEHOLDER | — | — | Yes | Not applicable |
| grape / healthy | PLACEHOLDER | — | — | Yes | Not applicable |
| peach / healthy | PLACEHOLDER | — | — | Yes | Not applicable |
| pepper,_bell / healthy | PLACEHOLDER | — | — | Yes | Not applicable |
| potato / healthy | PLACEHOLDER | — | — | Yes | Not applicable |
| raspberry / healthy | PLACEHOLDER | — | — | Yes | Not applicable |
| soybean / healthy | PLACEHOLDER | — | — | Yes | Not applicable |
| strawberry / healthy | PLACEHOLDER | — | — | Yes | Not applicable |
| tomato / healthy | PLACEHOLDER | — | — | Yes | Not applicable |
| apple / apple_scab | VERIFIED | https://blogs.cornell.edu/plantpathhvl/apple-diseases/apple-scab/ | Cornell University | Yes | No; source page unavailable |
| apple / black_rot | VERIFIED | https://blogs.cornell.edu/plantpathhvl/apple-diseases/black-rot/ | Cornell University | Yes | No; source page unavailable |
| apple / cedar_apple_rust | VERIFIED | https://extension.psu.edu/cedar-apple-rust | Penn State Extension | Yes | No; page unavailable |
| cherry_(including_sour) / powdery_mildew | VERIFIED | https://treefruit.wsu.edu/crop-protection/opm/cherry-powdery-mildew/ | Washington State University Extension | Yes | No; page unavailable |
| corn_(maize) / cercospora_leaf_spot gray_leaf_spot | VERIFIED | https://www.extension.purdue.edu/extmedia/BP/BP-56-W.pdf | Purdue Extension | Yes | No; page unavailable |
| corn_(maize) / common_rust_ | VERIFIED | https://www.extension.purdue.edu/extmedia/BP/BP-82-W.pdf | Purdue Extension | Yes | Yes, after replacing old prose with only the supported page 4 passage |
| corn_(maize) / northern_leaf_blight | VERIFIED | https://extension.entm.purdue.edu/fieldcropsipm/diseases/northern-corn-leaf-blight.php | Purdue Extension | Yes | No; page unavailable |
| grape / black_rot | VERIFIED | https://ohioline.osu.edu/factsheet/plpath-fru-24 | Ohio State University Extension | Yes | No; access denied |
| grape / esca_(black_measles) | VERIFIED | https://ipm.ucanr.edu/PMG/r302100511.html | UC IPM | Yes | Partly; page description and caution that wood infection cannot be eradicated by fungicides. Existing advice is not supported. |
| grape / leaf_blight_(isariopsis_leaf_spot) | VERIFIED | https://nrcgrapes.icar.gov.in/ | ICAR NRCG | Yes | No; generic homepage, no cited passage |
| orange / haunglongbing_(citrus_greening) | VERIFIED | https://crec.ifas.ufl.edu/research/citrus-production/disease-management/huanglongbing/ | University of Florida IFAS | Yes | No; page unavailable |
| peach / bacterial_spot | VERIFIED | https://extension.psu.edu/bacterial-spot-of-peach | Penn State Extension | Yes | No; page unavailable |
| pepper,_bell / bacterial_spot | VERIFIED | https://agritech.tnau.ac.in/crop_protection/crop_prot_crop_diseases_veg_chillies.html | Tamil Nadu Agricultural University | Yes | No; page unavailable |
| squash / powdery_mildew | VERIFIED | https://vegetablemdonline.ppath.cornell.edu/factsheets/Cucurbits_PM.htm | Cornell University | Yes | No; page unavailable |
| strawberry / leaf_scorch | VERIFIED | https://extension.psu.edu/strawberry-diseases-leaf-scorch | Penn State Extension | Yes | No; page unavailable |
| tomato / leaf_mold | VERIFIED | https://plantdisease.net/tomato_leaf_mold | Publisher claim not independently verified | Yes | No; page unavailable |
| tomato / septoria_leaf_spot | VERIFIED | https://www.extension.purdue.edu/extmedia/BP/BP-22-W.pdf | Purdue Extension | Yes | No; page unavailable |
| tomato / spider_mites two-spotted_spider_mite | VERIFIED | https://ipm.ucanr.edu/PMG/r783400111.html | UC IPM | Yes | No; redirects to russet mite |
| tomato / target_spot | VERIFIED | https://edis.ifas.ufl.edu/publication/PP316 | University of Florida IFAS | Yes | No; page unavailable |
| tomato / tomato_yellow_leaf_curl_virus | VERIFIED | https://ipm.ucanr.edu/PMG/r783101611.html | UC IPM | Yes | No; page unavailable |
| tomato / tomato_mosaic_virus | VERIFIED | https://vegetablemdonline.ppath.cornell.edu/factsheets/Virus_Tomato.htm | Cornell CALS | Yes | No; page unavailable |

## Manual source checks requested

Before any entry is marked VERIFIED, manually open its source URL and compare each retained sentence against the cited page. Start with the entries that will be shown in the demo: tomato early blight, tomato late blight, potato late blight, tomato bacterial spot, and corn common rust. For each, record the exact page title, publisher, passage heading/page, access date, and reviewer in `kb/advisories.yaml`. Do not restore removed dosages or product names unless the source passage directly contains them and local registration review is completed.

Weather rules currently have no cited source URLs. Until sources are added and verified, the API must report weather risk as unknown and must not apply the numeric rules.
