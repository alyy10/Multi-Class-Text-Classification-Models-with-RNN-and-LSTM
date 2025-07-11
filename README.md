# **Multi Class Text Classification using RNN and LSTM Models**

## **Project Overview**

This project focuses on building **multi-class text classification models** using **Recurrent Neural Networks (RNN)** and **Long Short-Term Memory (LSTM)** networks. The goal is to classify customer complaints about consumer financial products into different categories using their textual content. The dataset contains over two million complaints, and we use pre-trained word embeddings (GloVe) to map words to vector representations, which are then input to the models for training.

---

## **Business Overview**

Text classification is a fundamental task in **Natural Language Processing (NLP)**, with significant applications in many industries. By using deep learning methods like **RNN** and **LSTM**, this project aims to automate the categorization of customer complaints, which can improve customer support efficiency and help financial institutions address issues more effectively.

---

## **Learning Objectives**

- What **pre-trained word vectors** are and how to use them for text classification.
- The concept of **Recurrent Neural Networks (RNN)** and how they are applied to sequence data like text.
- The **vanishing gradient problem** and how **LSTM** solves it.
- How to **process GloVe embeddings** for use in deep learning models.
- Data preparation techniques like text **cleaning**, **tokenization**, and **indexing**.
- How to build **RNN** and **LSTM** models for text classification in **PyTorch**.
- Training models using **CUDA/CPU** and making predictions on new data.

---

## **Project Aim**

The primary aim of this project is to build and train **RNN** and **LSTM** models to perform **multi-class text classification** on a dataset containing customer complaints about financial products.

---

## **Dataset Description**

The dataset contains over two million **customer complaints** related to various **consumer financial products**. The key columns in the dataset include:

- **Text of the complaint**: The actual complaint text raised by customers.
- **Product**: The type of product related to the complaint (e.g., credit card, loan, etc.).

Pre-trained **GloVe word vectors** are used to convert words in the text into vector representations.

## **Tech Stack**

- **Language**: Python
- **Libraries**:
  - **pandas**: For data manipulation and processing.
  - **torch**: For building and training deep learning models (RNN and LSTM).
  - **nltk**: For tokenizing text data.
  - **numpy**: For numerical operations, especially handling embeddings.
  - **pickle**: For serializing models and data.
  - **re**: For text cleaning and preprocessing.
  - **tqdm**: For progress bars.
  - **sklearn**: For encoding labels and splitting the dataset.

---

## **Prerequisites**

1. **PyTorch** framework installed.
2. **GloVe embeddings** downloaded (50-dimensional vectors from the provided link).
3. Basic understanding of **RNNs**, **LSTMs**, and **word embeddings**.

---

## **Project Structure**

```
|-- Input/
|   |-- complaints.csv
|   |-- glove.6B.50d.txt
|
|-- Output/
|   |-- embeddings.pkl
|   |-- label_encoder.pkl
|   |-- labels.pkl
|   |-- model_lstm.pkl
|   |-- model_rnn.pkl
|   |-- vocabulary.pkl
|   |-- tokens.pkl
|
|-- Source/
|   |-- model.py
|   |-- data.py
|   |-- utils.py
|
|-- config.py
|-- Engine.py
|-- processing.py
|-- predict.py
|-- README.md
|-- requirements.txt
```

### **Folders and Files:**

1. **Input Folder**:

   - **`complaints.csv`**: The dataset containing customer complaints.
   - **`glove.6B.50d.txt`**: Pre-trained GloVe word embeddings (50-dimensional vectors for each word).
2. **Output Folder**:

   - **`embeddings.pkl`**: Processed word embeddings used in training.
   - **`labels.pkl`**: Encoded labels for classification.
   - **`label_encoder.pkl`**: Label encoder used to encode and decode labels.
   - **`model_lstm.pkl`**: Saved LSTM model after training.
   - **`model_rnn.pkl`**: Saved RNN model after training.
   - **`vocabulary.pkl`**: The vocabulary built from the dataset for tokenization.
   - **`tokens.pkl`**: Tokenized and indexed text data.
3. **Source Folder**:

   - **`model.py`**: Contains the model architecture for RNN and LSTM.
   - **`data.py`**: Handles data loading, cleaning, and tokenization.
   - **`utils.py`**: Helper functions such as saving and loading data.
4. **config.py**: Contains configuration settings (file paths, model parameters).
5. **Engine.py**: The main script for training the models. It loads data, builds the model, and trains it.
6. **processing.py**: Used for data preprocessing including cleaning, tokenizing, and indexing text.
7. **predict.py**: Used to make predictions on new complaint data using the trained models.
8. **requirements.txt**: Contains all the required libraries and their versions for setting up the environment.
9. **README.md**: This file that contains the instructions and overview of the project.

---

## **Solution Approach**

### **Step 1: Install Required Packages**

Install the necessary libraries by running:

```bash
pip install -r requirements.txt
```

### **Step 2: Import Required Libraries**

Import the essential libraries such as `pandas`, `torch`, `nltk`, etc.

### **Step 3: Define Configuration File Paths**

Configuration settings for paths like where the dataset, embeddings, and trained models are stored.

### **Step 4: Process GloVe Embeddings**

- Read the GloVe file.
- Convert the embeddings from strings to NumPy arrays.
- Add padding for special tokens (`<pad>` and `<unk>`).
- Save the embeddings and vocabulary for future use.

### **Step 5: Process Text Data**

- Clean the text (convert to lowercase, remove digits, and punctuation).
- Tokenize the text.
- Encode labels into numerical format using **LabelEncoder**.
- Save the processed tokens and labels.

### **Step 6: Build Data Loader**

- Use the `TextDataset()` function to convert the preprocessed text data and corresponding labels into a PyTorch dataset. A DataLoader is then used to create batches of data for efficient model training.

### **Step 7: Build the Models (RNN & LSTM)**

- **RNN Model**: Use `torch.nn.RNN` for the RNN model architecture.
- **LSTM Model**: Use `torch.nn.LSTM` for the LSTM model architecture.

### **Step 8: Model Training**

- Train both the RNN and LSTM models using **CrossEntropyLoss** and the **Adam optimizer**.
- Use GPU (CUDA) or CPU for model training.

### **Step 9: Predictions**

- Once the models are trained, make predictions on new customer complaint data using the trained models.

---

## **Project Takeaways**

By completing this project, you will gain experience in:

1. Using **pre-trained word vectors** (GloVe) for text classification.
2. Understanding and implementing **RNNs** and **LSTMs** for sequential data.
3. **Data preprocessing** techniques like tokenization, cleaning, and encoding.
4. Training models on **GPU** (CUDA) or **CPU**.
5. Making **predictions** using deep learning models on text data.

---

## **Conclusion**

This project demonstrates how to use **RNN** and **LSTM** models to classify customer complaints into predefined categories. By implementing these models in PyTorch and processing the data effectively, you can automate text classification tasks for a variety of real-world applications.# Multi-Class-Text-Classification-Models-with-RNN-and-LSTM
