# Prototypical Transfer Learning for Ancient Script Decipherment

This repository contains a specialized **Few-Shot Learning** framework designed to recognize and analyze ancient writing systems, such as the **Indus Valley Script (IVC)**. By utilizing Prototypical Networks, the model can generalize to new, unseen characters with minimal training data.

## 🚀 Key Features
* **Prototypical Networks:** Learns a metric space where classification is performed by computing distances to prototype representations of each class.
* **Transfer Learning:** Pre-trained on the Omniglot dataset to understand general character strokes before fine-tuning on ancient scripts.
* **Ancient Script Support:** Specifically optimized for low-resource datasets where traditional Deep Learning (CNNs/ViTs) fails due to data scarcity.

## 📂 Repository Structure
* `app.py`: Main entry point for inference and script recognition.
* `explore_dataset.ipynb`: Data exploration and visualization of the script clusters.
* `indus_valley_few_shot_model.pth`: The trained model weights for the Prototypical Network.

## 🛠️ Getting Started
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Mahesh-Paul/Prototypical-Transfer-Learning-Framework-for-Ancient-Script-Deciphermen.git](https://github.com/Mahesh-Paul/Prototypical-Transfer-Learning-Framework-for-Ancient-Script-Deciphermen.git)