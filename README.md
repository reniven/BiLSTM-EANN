# Fake News Detection with Adversarial Networks (BiLSTM-EANN)

## Overview
Social media has amplified the spread of **fake news**, which often travels faster and deeper than real news. Manual detection is infeasible, so this project explores **deep learning and adversarial networks** for detecting fake news, focusing on **newly emerging events** where traditional models fail.

Inspired by [EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection](https://dl.acm.org/doi/10.1145/3219819.3219903) and leveraging **BERT, ResNet-18, and CNN-based feature extractors**, our approach fuses **text, images, and social context** for robust, multimodal detection.

---

## Problem
- Existing models depend heavily on event-specific features and fail on unseen events.  
- Fake news evolves quickly, requiring **generalizable representations**.  

**Our solution:**  
Train an adversarial network that extracts **event-invariant features** while classifying news as real or fake.  

---

## Datasets
- **MediaEval 2016**: ~13k Twitter posts (text, images, social features), binary classification.  
- **LIAR**: 12.8k statements from Politifact, fine-grained labels (True → Pants-fire).  

---

## Model
- **Text**: BERT  
- **Images**: ResNet-18  
- **Social context**: CNN-based extractor  
- **Detector**: CNN & AC-BiLSTM  
- **Event Discriminator**: Adversarial training for generalization  

---

## Results
- Training/validation accuracy: ~60% on MediaEval & LIAR.  
- Early tests show **event-invariant features help generalize**, but loss stagnation requires further debugging.  

# Authors & Contributors

This research project was written and contributed by Eric Le, Xinyan Xie, Jianan Liu, and Qiwei Ge.
