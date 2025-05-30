# Disease Classification with TF-IDF and One-Hot Encoding

## Overview
This project focuses on classifying diseases based on their associated risk factors, symptoms, and signs. It employs two feature extraction techniques: TF-IDF and one-hot encoding, to evaluate their effectiveness in disease classification.

## Tasks
### Task 1: TF-IDF Feature Extraction
1. Parse Risk Factors, Symptoms, and Signs from stringified lists to Python lists.
2. Convert these lists into single strings (e.g., `["fever", "stress"]` → `"fever stress"`).
3. Apply TF-IDF vectorization using `TfidfVectorizer` from `sklearn.feature_extraction.text`.
4. Combine the TF-IDF matrices into a single feature matrix.
5. Compare the TF-IDF matrix with a provided one-hot encoded matrix in terms of sparsity and unique features.

### Task 2: Dimensionality Reduction
1. Apply PCA and Truncated SVD to both TF-IDF and one-hot encoded matrices, reducing dimensions to 2-3 components.
2. Visualize the reduced dimensions using 2D plots, color-coded by disease category (e.g., cardiovascular, neurological).
3. Discuss which encoding method produces more separable clusters.

### Task 3: Model Training
1. Train KNN models with k = 3, 5, 7 using:
   - Euclidean
   - Manhattan
   - Cosine similarity
2. Perform 5-fold cross-validation for each k and metric combination, reporting:
   - Accuracy
   - Precision
   - Recall
   - F1-score
3. Train Logistic Regression models on both matrices and compare performance against KNN.
4. Compare results across:
   - Encoding methods (TF-IDF vs. one-hot)
   - Distance metrics (for KNN)
   - Model types (KNN vs. Logistic Regression)

### Task 4: Critical Analysis
- Discuss why TF-IDF might outperform one-hot encoding (or vice versa).
- Evaluate the clinical relevance of the results and whether TF-IDF clusters align with real-world disease categories.
- Identify limitations of both encoding methods.

