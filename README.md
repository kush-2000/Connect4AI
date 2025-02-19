# **🤖 Connect4AI: A Machine Learning-Powered Connect 4 Bot**

## **📌 Overview**
Connect4AI is a full-stack project that trains a **Convolutional Neural Network (CNN)** and a **Transformer model** to play **Connect 4**. The dataset is generated using **Monte Carlo Tree Search (MCTS)**, and the AI is deployed on **AWS** using **Docker**. A web interface built with **Anvil** allows users to play against the AI.

---

## **🚀 Features**
✔️ **AI-powered Connect 4 gameplay** using CNN and Transformer models  
✔️ **Monte Carlo Tree Search (MCTS)** for data generation  
✔️ **Interactive web app** using Anvil  
✔️ **Backend hosted on AWS Lightsail**  
✔️ **Dockerized deployment** for scalability  
✔️ **Performance comparison** between CNN and Transformer models  

---

## **🌍 Website Access**
🔗 **Website**: [Connect4AI Web App](https://ambitious-grim-station.anvil.app/)  

🔑 **Login Credentials**  
- **Email**: `dan`  
- **Password**: `Optimization1234`  

---

## **🛠 Tech Stack**
🔹 **Machine Learning**: TensorFlow, Keras, NumPy, Pandas  
🔹 **Backend**: Python, Anvil Uplink, Flask API, AWS Lightsail  
🔹 **Frontend**: Anvil Web Framework  
🔹 **Deployment**: Docker, Docker Compose  

---

## **⚡ Setup & Installation**
### **1️⃣ Clone the Repository**
- Clone the repository from GitHub and navigate into the project directory.  

### **2️⃣ Install Dependencies**
- Install all required dependencies from the provided requirements file.  

### **3️⃣ Run the Model Locally**
- Execute the model script to test it locally before deployment.  

### **4️⃣ Deploy to AWS**
- Set up an **AWS Lightsail instance**, transfer necessary files, and run the **Docker container** for deployment.  

---

## **📊 Data Generation**
Training data is generated using **Monte Carlo Tree Search (MCTS)**:  
🔹 **Self-play**: AI plays against itself to determine optimal moves  
🔹 **Randomized starts**: Some games begin with random moves before MCTS takes over  
🔹 **Duplicate filtering**: Selects the most frequently recommended move  
🔹 **Final dataset size**: **300,000+ board states**  

### **🔍 Challenges**
⏳ **Long runtime** requiring multiple overnight runs  
🎲 **MCTS randomness** leading to variations in move recommendations  

---

## **🤖 Model Training**
### **🧠 CNN Architecture**
🔸 **Input**: (6,7,2) board representation  
🔸 **Convolutional layers** extract spatial features  
🔸 **Batch normalization** ensures stable training  
🔸 **Fully connected layers** process extracted features for decision-making  
🔸 **Softmax activation** determines the best move  

### **🔀 Transformer Architecture**
🔹 **Converts board states into sequences** for processing  
🔹 **Utilizes self-attention layers** to understand board relationships  
🔹 **Layer normalization** maintains stability  
🔹 **Multi-head attention** improves learning of key game positions  
🔹 **Final output determines** the best move for the AI  

### **⚙️ Training Optimization**
✅ **Early stopping** prevents overfitting  
🔧 **Learning rate tuning** improves convergence speed  
🏋️ **Batch size adjustments** ensure smooth gradient updates  

---

## **🎮 Web Interface**
The web app is built with **Anvil** and allows users to:  
1️⃣ **Log in** with predefined credentials  
2️⃣ **Play against the AI**, choosing between the CNN and Transformer models  
3️⃣ **Track game results** and analyze AI performance  

### **🌍 Hosting**
🔸 The **Anvil frontend** communicates with the **backend API** hosted on AWS  
🔸 AI models process game states and return the best move in real time  

---

## **☁️ Backend Deployment**
### **1️⃣ AWS Lightsail Setup**
✔️ An **AWS Lightsail instance** is created to host the backend  
✔️ Required files and trained models are uploaded for deployment  

### **2️⃣ Dockerization**
✔️ The **backend is packaged into a Docker container** for consistency and scalability  
✔️ The container is deployed to AWS, enabling real-time interactions with the web app  

---

## **📈 Results**
### **🏆 CNN Performance**
🎯 **Training Accuracy**: **74.2%**  
🎯 **Validation Accuracy**: **65.17%**  
🏅 **Strength**: Strong tactical play and effective move prediction  

### **⚠️ Transformer Performance**
📉 **Training Accuracy**: **59.9%**  
📉 **Validation Accuracy**: **54.3%**  
❌ **Weakness**: Struggles with recognizing threats and defensive moves  

### **🆚 AI Gameplay Evaluation**
🏆 The **CNN model consistently wins** against human players  
❗ The **Transformer model struggles** in certain strategic situations  

---

## **🚀 Future Improvements**
🔍 **Improve Transformer model** through hyperparameter tuning  
🤖 **Train AI using reinforcement learning** with self-play  
📊 **Implement a leaderboard** to track user performance  
🏗️ **Analyze AI move efficiency** to refine strategy  

---

## **👨‍💻 Contributors**
- **Kush Patel** (ksp946)  
- **Shirley Liu** (xl22445)  
- **Samuel Chen** (swc872)  
- **Ronak Goyal** (rg49395)  
