
# Group 8 ArewaDS Capstone Project
------------------------------------------------------------------------------------ 
## ArewaDS Deep Learning with PyTorch Cohort 2.0
------------------------------------------------------------------------------------

# Improving Binary Classification of Tomato Leaf Health in Northern Nigeria Using Transfer Learning on TomatoEbola Dataset

![ArewaDS Logo](ArewaDS_logo.PNG)

## Project Overview

This capstone project, developed under the **Arewa Data Science Academy - Deep Learning Fellowship Cohort 2**, aims to improve binary classification (Healthy vs. Infected) of tomato leaf health using **transfer learning techniques**. The work builds on the recent publication: ["Early detection of tomato leaf diseases using transformers and transfer learning" by Shehu et al. (2025)](https://www.sciencedirect.com/science/article/pii/S1161030125001212), and seeks to overcome its limitations through deeper experimentation, dataset enhancements, and practical deployment.

## Objective

To explore and compare the performance of four state-of-the-art pre-trained models (ResNet, MobileNet, EfficientNet, and Vision Transformer - ViT) for classifying tomato leaves into healthy or infected categories. The goal is to identify the best-performing models and deploy them via a Gradio web interface for real-time prediction.

---

## Dataset Description

The project uses an enhanced and localized version of the TomatoEbola dataset, which contains labeled images of tomato leaves collected from three ward farms in Northern Nigeria. The dataset is organized as follows:

```
|- KUKARETA WARD FARM
|   |-- Infected
|   |-- Healthy
|
|- KASAISA WARD FARM
|   |-- Infected
|   |-- Healthy
|
|- DIKUMARI WARD FARM
|   |-- Infected
|   |-- Healthy
|
|- VillagePlant
    |-- Infected
    |-- Healthy
```

> The **VillagePlant** dataset is a merged version of all three ward-level datasets and serves as the main dataset for training the final models.

---

## Models and Methodology

We used four pre-trained convolutional and transformer-based models:

* **ResNet**
* **MobileNet**
* **EfficientNet**
* **Vision Transformer (ViT)**

### Steps Taken:

1. **Data Cleaning & Preprocessing**: Each image was resized and normalized. Augmentations like rotation, flipping, and color jittering were applied to prevent overfitting.
2. **Training**: Each model was trained independently on the merged VillagePlant dataset.
3. **Evaluation**:

   * Model Accuracy
   * Loss Curve
   * Confusion Matrix
   * Grad-CAM Visualization for interpretability
4. **Deployment**: The best-performing models were deployed using [Gradio](https://www.gradio.app/).

### Performance Summary on VillagePlant Dataset

| Model        | Accuracy | Loss   |
| ------------ | -------- | ------ |
| ViT          | 100%     | 0.07   |
| EfficientNet | 96.7%    | 0.16   |
| ResNet       | \~90%    | \~0.30 |
| MobileNet    | \~89%    | \~0.34 |

> Based on the results, **ViT** and **EfficientNet** were selected for deployment.

---

## Deployment Highlights

The final Gradio web app integrates the following features:

* Tomato leaf image upload
* Model selection (ViT or EfficientNet)
* Prediction label: **Infected** or **Healthy**
* Grad-CAM heatmap overlay
* Confidence scores and class probabilities

![Deployed Screenshot](Screenshot%20\(32\).png)

---

## Visualizations & Explainability

* **Grad-CAM** was implemented to interpret the regions of the leaf that influenced model predictions.
* **Confusion Matrix** was used to assess true/false positive and negative classifications.

---

## Experiment Tracking

We used [Weights & Biases (wandb)](https://wandb.ai/) to log:

* Training metrics
* Loss/accuracy curves
* Parameter summaries

---

## Contributors

**Team Name:** Group 8 - ArewaDS Capstone Project
**Mentor:** Engr. Bala Abduljalil
**Team Members:**

* Abubakar Abubakar Al-amin
* Bernard Adjei Buckman
* Halimat Musa
* Kaloma Usman Majikumna

---

## Citation

Shehu, H. A., Ackley, A., Mark, M., & Eteng, E. O. (2025). *Early detection of tomato leaf diseases using transformers and transfer learning*. Computers and Electronics in Agriculture. [https://www.sciencedirect.com/science/article/pii/S1161030125001212](https://www.sciencedirect.com/science/article/pii/S1161030125001212)

---

## License

This project is made available for educational purposes and free community use under the ArewaDS Capstone Initiative.

---

## Acknowledgements

Special thanks to **Arewa Data Science Academy**, our mentor, and all team members for their collaboration, resilience, and technical contributions.

---

## How to Run the Project

1. Clone the GitHub repo (if available)
2. Load the notebook `Main.ipynb`
3. Run all cells to retrain or test models
4. For deployment: run `App.py` to launch Gradio interface

> The ArewaDS logo and screenshots are embedded in the notebook and Gradio interface.

---

For questions or feedback, reach out via: [deeplearningfellowship@gmail.com](mailto:deeplearningfellowship@gmail.com)


