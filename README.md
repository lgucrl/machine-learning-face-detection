# Machine Learning face detection

This project implements a **lightweight face detection pipeline** aimed at running with **limited computing capacity** by combining classic computer vision methods with a fast, well-established machine learning model. Instead of deep neural networks, the project uses **Histogram of Oriented Gradients (HOG)** features and a **Support Vector Classifier (SVC)** to distinguish *face* vs *non-face* image patches. For detection in full-size images, the trained classifier is applied using a **sliding-window** approach and refined with **Non-Maximum Suppression (NMS)** to produce final bounding boxes.

---

## Datasets

The training data is built as a **binary classification dataset** by combining two public image sources:

- **[UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new) (positive class - faces):** a large collection of face images (200×200) covering wide variation in age, pose, expression, and lighting. For this project, a random subset is sampled to represent the “face” class.
- **[Caltech 256](https://www.kaggle.com/datasets/jessicali9530/caltech256) (negative class - non-faces):** a diverse object dataset with 257 categories and >30,000 images of varying size. Categories likely to contain faces/people (e.g., `159.people`, `205.superman`, `253.faces-easy-101`) are excluded before sampling “non-face” examples.

A balanced dataset of **10,000** images is created by sampling **5,000 faces** and **5,000 non-faces**, resizing each image to **128×128** and saving them into separate folders such as `face/` and `non_face/`.

---

## Project workflow

1. **Data collecting, inspection, and dataset assembly**  
   The workflow starts by downloading/collecting candidate images for both classes and inspecting them to verify variety and quality. The key goal is to construct a clean “non-face” set: Caltech 256 categories that can include faces or people are excluded to avoid “hidden positives” that would confuse the classifier. Next, a balanced sample is created (equal faces and non-faces) and all images are resized to a fixed resolution (128×128) so the feature extractor and classifier operate on a consistent input shape.

2. **Extracting HOG descriptors**  
   Each image is converted to grayscale and transformed using **HOG**, which captures local edge direction and shape features that are particularly informative for faces. The implementation uses a configuration of 9 orientation bins, 8×8 pixels per cell, 2×2 cells per block and L2-Hys normalization, producing a fixed-length vector (8,100 features per image). This step turns raw pixels into a feature matrix `X` and labels into a vector `y`.

3. **Train/test split with stratification**  
   The dataset is divided into training and test subsets (with 80/20 ratio) using **stratification** so the 1:1 class balance (face/non-face) remains consistent in both sets. This prevents misleading results and that reported metrics are inflated due to class imbalance.

4. **Model training and hyperparameter optimization**  
   Training uses an **SVC** inside a Scikit-learn **Pipeline** that includes a **StandardScaler**, since SVMs are sensitive to feature scale. Hyperparameters (`C`, kernel type, and `gamma` when using an RBF kernel) are tuned with cross-validation via randomized search to efficiently explore the parameter space. The result is a small model that is fast at inference, performs strongly on HOG features, and can output probabilities when enabled.

5. **Evaluation and model selection**  
   The trained models are evaluated  on the test set using classification metrics (accuracy, precision, recall, F1) and threshold-based curves (ROC and Precision–Recall). Confusion matrices are inspected to identify whether errors are dominated by false positives (non-faces classified as faces) or false negatives (missed faces). This analysis informs whether a linear or RBF kernel provides the best trade-off between speed and accuracy.

6. **Full-image face detection with sliding windows and NMS**  
   Lastly, a pipeline to detect faces in full-size images is implemeted, based on a **128×128 sliding window** scanning across the input image at a chosen stride. Each window is scored by the trained classifier and windows above a probability threshold become candidate detections. Because many adjacent windows often overlap around the same face, **Non-Maximum Suppression** uses IoU to keep only the highest-confidence bounding boxes. The final output is a list of bounding boxes for detected faces that can be visualized directly on the original image with confidence scores.

---

## Tech stack

- Python
- OpenCV (image I/O and visualization)
- Scikit-learn (Pipeline, SVC, hyperparameter search)
- HOG feature extraction (skimage implementation)
