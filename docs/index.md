# AniMask
## Anime-Informed Masking for Masked Autoencoders

Team:
- Tingting Du
- Frank Sun
- Xin Chen
- Minyuan Zhu

---

## Motivation
Anime images exhibit strong semantic structure (faces, eyes, hairstyles) and flat backgrounds,
making standard MAE assumptions unsuitable.

---

## Project Proposal
Our original goal was to design semantic-aware masking strategies:

- Attention-guided masking (S1)
- Foreground-background masking (S2)
- Part-aware curriculum masking (S3)

[Download Proposal (PDF)](path_to_pdf)

---

## Midterm Results

### Reconstruction Quality
<img src="assets/images/recon1.png" width="600" />

Observations:
- Severe block artifacts early
- Color inconsistency
- Strong improvement after long training

---

### Mask Ratio Study

| Mask Ratio | Result |
|-------------|--------|
| 25%         | Best details |
| 50%         | Balanced |
| 75%         | Blurry |
| 90%         | Abstract |

---

## Proposal Revision
Due to compute limits, we pivoted from training-from-scratch to:

- Fine-tuning official MAE
- Anime face recognition benchmark

---

## Technical Approach

Pipeline:
