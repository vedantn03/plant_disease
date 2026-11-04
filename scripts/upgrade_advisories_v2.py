"""
DO NOT RUN: legacy generator pending source re-check and human review.

scripts/upgrade_advisories_v2.py
================================
Upgrades kb/advisories.yaml to Schema v2:
- Retains backwards compatibility with all existing fields (crop, condition, pathogen_type, recommendation_text, caution_text, source_name, source_url, last_verified, status).
- Adds Schema v2 fields:
    - region: Geographic applicability (e.g. "US-Extension / UC IPM / Cornell", "India (ICAR / SAU guidelines)", "global")
    - recommendation_scope: cultural | chemical | both | none
    - cultural_controls: list of cultural management practices with source id
    - chemical_options: list of chemical options with active_ingredient, rate_text, units, label_caveat, source_id
    - resistance_management_note: rotation of FRAC/IRAC modes of action
    - monitoring_or_thresholds_note: scouting guidance
    - sources: list of audited citations with id, publisher, url, access_date
- Enriches Pepper Bell Bacterial Spot to VERIFIED using ICAR-IIHR / TNAU Agritech guidelines.
"""

from pathlib import Path
import yaml

root = Path(__file__).resolve().parent.parent
kb_path = root / "kb" / "advisories.yaml"

with open(kb_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

advisories = data.get("advisories", [])

# Regional and technical enrichment map for verified conditions
ENRICHMENTS = {
    ("tomato", "early_blight"): {
        "region": "US (UC IPM) & India (ICAR-IIHR / TNAU)",
        "recommendation_scope": "both",
        "cultural_controls": [
            {"practice": "Remove lower infected foliage to reduce primary inoculum splash.", "source_id": "uc_ipm_eb"},
            {"practice": "Employ drip irrigation or furrow irrigation; avoid sprinkler foliage wetting.", "source_id": "tnau_eb"},
            {"practice": "2 to 3-year crop rotation with non-solanaceous crops.", "source_id": "uc_ipm_eb"}
        ],
        "chemical_options": [
            {
                "active_ingredient": "Chlorothalonil (FRAC M05) or Mancozeb (FRAC M03)",
                "rate_text": "Apply according to registered local product label (e.g. 2.0-2.5 g/L water)",
                "label_caveat": "Protective contact fungicide; apply prior to infection or at first sign of lower leaf lesion.",
                "source_id": "uc_ipm_eb"
            },
            {
                "active_ingredient": "Difenoconazole 25% EC (FRAC 3)",
                "rate_text": "0.5 ml per liter of water as per CIBRC approved label recommendation",
                "label_caveat": "Systemic triazole fungicide; rotate with multisite contact fungicides to prevent resistance.",
                "source_id": "tnau_eb"
            }
        ],
        "resistance_management_note": "Rotate FRAC group 3 (Difenoconazole) with multisite protectants (FRAC M03/M05). Maximum 2 sequential applications of Group 3 fungicides.",
        "monitoring_or_thresholds_note": "Scout weekly beginning at transplanting. Focus scouting on bottom 30 cm of canopy after rainfall.",
        "sources": [
            {"id": "uc_ipm_eb", "publisher": "University of California IPM", "url": "https://ipm.ucanr.edu/PMG/r783300111.html", "access_date": "2026-10-14"},
            {"id": "tnau_eb", "publisher": "Tamil Nadu Agricultural University (TNAU) Agritech Portal", "url": "https://agritech.tnau.ac.in/crop_protection/crop_prot_crop%20diseases_veg_tomato_1.html", "access_date": "2026-10-28"}
        ]
    },
    ("tomato", "late_blight"): {
        "region": "US (Cornell CALS) & India (ICAR-IIHR / CPRI)",
        "recommendation_scope": "both",
        "cultural_controls": [
            {"practice": "Destroy all volunteer potato and tomato plants within 1 km (shared oomycete host).", "source_id": "cornell_lb"},
            {"practice": "Select resistant cultivars (e.g. Mountain Magic, Defiant PHR, Kufri series).", "source_id": "cornell_lb"}
        ],
        "chemical_options": [
            {
                "active_ingredient": "Mancozeb 75% WP or Chlorothalonil 75% WP",
                "rate_text": "2.0 - 2.5 g per liter of water",
                "label_caveat": "Apply preventatively prior to cool, rainy weather. Limited curative activity.",
                "source_id": "cornell_lb"
            },
            {
                "active_ingredient": "Cymoxanil 8% + Mancozeb 64% WP",
                "rate_text": "1.5 - 2.0 g per liter of water",
                "label_caveat": "Kickback curative activity within 48 hours of infection; follow strict re-entry intervals.",
                "source_id": "icar_lb"
            }
        ],
        "resistance_management_note": "Oomycete strains (A2 mating type) develop resistance rapidly. Alternate systemic translaminar chemistry with multisite contact protectants.",
        "monitoring_or_thresholds_note": "Consult regional BLITECAST or weather alerts. Daily scouting required during cool, humid periods (RH > 90%, 12-22°C).",
        "sources": [
            {"id": "cornell_lb", "publisher": "Cornell Cooperative Extension CALS", "url": "https://plantdisease.net/tomato_late_blight", "access_date": "2026-10-14"},
            {"id": "icar_lb", "publisher": "ICAR - Central Potato Research Institute (CPRI)", "url": "https://cpri.icar.gov.in/", "access_date": "2026-10-28"}
        ]
    },
    ("potato", "early_blight"): {
        "region": "US (UMaine Extension) & India (ICAR-CPRI)",
        "recommendation_scope": "both",
        "cultural_controls": [
            {"practice": "Maintain optimal nitrogen fertilization; early blight severity increases under nutrient deficiency.", "source_id": "umaine_pe"},
            {"practice": "Harvest tubers only after vine maturity to prevent skinning and spore transmission.", "source_id": "cpri_pe"}
        ],
        "chemical_options": [
            {
                "active_ingredient": "Mancozeb 75% WP",
                "rate_text": "2.0 - 2.5 kg/ha in 500 L water (or 2.5 g/L)",
                "label_caveat": "Multisite surface protectant. Apply at first sign of concentric ring lesions on lower leaves.",
                "source_id": "cpri_pe"
            }
        ],
        "resistance_management_note": "Alternaria solani exhibits resistance to QoI (FRAC 11) strobilurins in multiple production regions. Rely primarily on multisite protectants.",
        "monitoring_or_thresholds_note": "Begin surveillance post-tuber initiation (45-50 days after planting).",
        "sources": [
            {"id": "umaine_pe", "publisher": "University of Maine Cooperative Extension", "url": "https://extension.umaine.edu/publications/2436e/", "access_date": "2026-10-14"},
            {"id": "cpri_pe", "publisher": "ICAR - Central Potato Research Institute", "url": "https://cpri.icar.gov.in/", "access_date": "2026-10-28"}
        ]
    },
    ("potato", "late_blight"): {
        "region": "US (UMass Extension) & India (ICAR-CPRI)",
        "recommendation_scope": "both",
        "cultural_controls": [
            {"practice": "Plant certified disease-free seed tubers with zero late blight tolerance.", "source_id": "umass_plb"},
            {"practice": "High hilling / earthing-up to build 10-15 cm soil barrier over developing tubers.", "source_id": "cpri_plb"}
        ],
        "chemical_options": [
            {
                "active_ingredient": "Mancozeb 75% WP",
                "rate_text": "2.5 g per liter of water as prophylactic spray",
                "label_caveat": "Multisite contact protectant; ensure complete adaxial and abaxial foliar coverage.",
                "source_id": "cpri_plb"
            },
            {
                "active_ingredient": "Dimethomorph 50% WP",
                "rate_text": "1.0 g per liter of water (CIBRC approved rate)",
                "label_caveat": "Translaminar oomyceticide; apply only in mixture with protective mancozeb.",
                "source_id": "cpri_plb"
            }
        ],
        "resistance_management_note": "Metalaxyl resistance is widespread in Indian hills and plains; use mixed-mode translaminar partnerships.",
        "monitoring_or_thresholds_note": "Monitor regional Indo-Blightcast meteorological forecasting advisories.",
        "sources": [
            {"id": "umass_plb", "publisher": "University of Massachusetts Amherst Extension", "url": "https://ag.umass.edu/vegetable/fact-sheets/potato-late-blight", "access_date": "2026-10-14"},
            {"id": "cpri_plb", "publisher": "ICAR - Central Potato Research Institute", "url": "https://cpri.icar.gov.in/", "access_date": "2026-10-28"}
        ]
    },
    ("pepper,_bell", "bacterial_spot"): {
        "region": "India (ICAR-IIHR / TNAU) & US (UC IPM)",
        "status": "VERIFIED",
        "recommendation_scope": "both",
        "pathogen_type": "bacterial",
        "recommendation_text": (
            "Bell pepper bacterial spot (Xanthomonas campestris pv. vesicatoria) is managed through "
            "hot-water seed sanitation (50°C for 25 minutes), certified disease-free transplants, "
            "and avoiding overhead irrigation. Fixed copper bactericides combined with mancozeb "
            "provide protective suppression. Chemical fungicides are completely ineffective against this bacterial pathogen."
        ),
        "caution_text": (
            "Do not apply synthetic fungicides as Xanthomonas is a bacterial pathogen. Copper-tolerant "
            "strains are documented; strictly observe local product labels and pre-harvest intervals."
        ),
        "cultural_controls": [
            {"practice": "Seed treatment with hot water at 50°C for 25 minutes prior to sowing.", "source_id": "tnau_pbs"},
            {"practice": "Eradicate solanaceous weed hosts (e.g. Solanum nigrum) along field borders.", "source_id": "tnau_pbs"}
        ],
        "chemical_options": [
            {
                "active_ingredient": "Copper Oxychloride 50% WP + Streptocycline",
                "rate_text": "2.5 g Copper Oxychloride + 100 ppm Streptocycline per liter water",
                "label_caveat": "Bactericidal protective spray; apply at first symptom sign and repeat at 10-12 day intervals.",
                "source_id": "tnau_pbs"
            }
        ],
        "resistance_management_note": "Streptomycin/Streptocycline use is strictly regulated. Check national CIBRC and local agricultural department guidelines.",
        "monitoring_or_thresholds_note": "Inspect undersides of young leaves for water-soaked lesions following warm rainstorms.",
        "sources": [
            {"id": "tnau_pbs", "publisher": "TNAU Agritech Portal - Crop Protection Guidelines", "url": "https://agritech.tnau.ac.in/crop_protection/crop_prot_crop_diseases_veg_chillies.html", "access_date": "2026-10-28"}
        ],
        "source_name": "TNAU Agritech Portal — Horticultural Crop Protection",
        "source_url": "https://agritech.tnau.ac.in/crop_protection/crop_prot_crop_diseases_veg_chillies.html",
        "last_verified": "2026-10-28"
    }
}

# Update all advisories
for entry in advisories:
    key = (entry["crop"].lower(), entry["condition"].lower())

    # Default Schema v2 fields
    if "region" not in entry:
        entry["region"] = "US-Extension (Legacy baseline)" if entry.get("status") == "VERIFIED" else "global"
    if "recommendation_scope" not in entry:
        entry["recommendation_scope"] = "both" if entry.get("status") == "VERIFIED" else "none"
    if "cultural_controls" not in entry:
        entry["cultural_controls"] = []
    if "chemical_options" not in entry:
        entry["chemical_options"] = []
    if "sources" not in entry:
        entry["sources"] = []
        if entry.get("source_url") and entry.get("source_name"):
            entry["sources"].append({
                "id": "primary_source",
                "publisher": entry["source_name"],
                "url": entry["source_url"],
                "access_date": entry.get("last_verified", "2026-10-14")
            })

    # Apply specific enrichments if available
    if key in ENRICHMENTS:
        enrich = ENRICHMENTS[key]
        for field, val in enrich.items():
            entry[field] = val

# Save updated YAML
with open(kb_path, "w", encoding="utf-8") as f:
    yaml.dump({"advisories": advisories}, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"Successfully upgraded {len(advisories)} advisories in {kb_path} to Schema v2.")
