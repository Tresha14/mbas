# Bias check

Data: synthetic test participants (person-wise split); rerun after our sessions are labelled. Groups with fewer than 20 items are omitted. A gap above 5.0 points between groups is flagged.

## Communication-effectiveness error (MAE, points)

**gender** - max gap 0.39

| group | n | mean_pred | mean_label | mae |
| --- | ---: | ---: | ---: | ---: |
| female | 176 | 60.55 | 58.43 | 5.63 |
| male | 297 | 59.57 | 56.26 | 6.02 |

**accent** - max gap 1.04

| group | n | mean_pred | mean_label | mae |
| --- | ---: | ---: | ---: | ---: |
| gujarati_english | 143 | 57.98 | 54.51 | 6.08 |
| hindi_english | 132 | 59.76 | 57.57 | 5.91 |
| neutral_indian_english | 88 | 56.01 | 52.99 | 6.28 |
| tamil_english | 110 | 65.84 | 63.03 | 5.24 |

**glasses** - max gap 0.47

| group | n | mean_pred | mean_label | mae |
| --- | ---: | ---: | ---: | ---: |
| False | 275 | 60.63 | 58.06 | 5.68 |
| True | 198 | 58.98 | 55.69 | 6.15 |

## Engagement accuracy (%)

**gender** - max gap 1.2

| group | n | engagement_accuracy_pct |
| --- | ---: | ---: |
| female | 2560 | 75.55 |
| male | 4288 | 76.75 |

**accent** - max gap 6.6 FLAGGED

| group | n | engagement_accuracy_pct |
| --- | ---: | ---: |
| gujarati_english | 2059 | 73.87 |
| hindi_english | 1895 | 80.47 |
| neutral_indian_english | 1327 | 74.53 |
| tamil_english | 1567 | 75.94 |

**glasses** - max gap 3.63

| group | n | engagement_accuracy_pct |
| --- | ---: | ---: |
| False | 4014 | 77.8 |
| True | 2834 | 74.17 |

## Gaze error per held-out MPIIGaze person (deg)

{'note': 'gaze not evaluated'}

## Notes

- The synthetic simulator generates traits independently of demographics, so gaps there mostly measure noise; the real test is our recordings (accent and glasses are recorded in meta.json).
- Engagement and attention from webcams can disadvantage people with disabilities, different eye-contact norms or poor cameras; follow-up flags stay explainable and decisions stay with the interviewer.
