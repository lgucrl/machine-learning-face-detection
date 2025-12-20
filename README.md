# Machine Learning face detection

This project implements a **lightweight face detection pipeline** aimed at running with **limited computing capacity** by combining classic computer vision methods with a fast, well-established machine learning model. Instead of deep neural networks, the project uses **Histogram of Oriented Gradients (HOG)** features and a **Support Vector Classifier (SVC)** to distinguish *face* vs *non-face* image patches. For detection in full-size images, the trained classifier is applied using a **sliding-window** approach and refined with **Non-Maximum Suppression (NMS)** to produce final bounding boxes.

---

## Datasets

The training data is built as a **binary classification dataset** by combining two public image sources:

- **UTKFace (positive class - faces):** a large collection of face images (200×200) covering wide variation in age, pose, expression, and lighting. For this project, a random subset is sampled to represent the “face” class.
- **Caltech 256 (negative class - non-faces):** a diverse object dataset with 257 categories and >30,000 images of varying size. Categories likely to contain faces/people (e.g., `159.people`, `205.superman`, `253.faces-easy-101`) are excluded before sampling “non-face” examples.

A balanced dataset of **10,000** images is created by sampling **5,000 faces** and **5,000 non-faces**, resizing each image to **128×128** and saving them into separate folders such as `face/` and `non_face/`.

---

## Project workflow
