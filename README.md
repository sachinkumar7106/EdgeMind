# EdgeMind: Deploying Transformer-Level Intelligence on Resource-Constrained Devices

EdgeMind explores whether a lightweight Convolutional Neural Network (CNN) can learn from a Vision Transformer (ViT) and approach its classification performance while using fewer parameters.

The project uses **knowledge distillation** on the **CIFAR-10** dataset. A Vision Transformer acts as the teacher model, while MobileNetV3-Small acts as the student model. The trained student is exposed through a Flask REST API and a React-based web interface.

## Table of Contents

- [Project Overview](#project-overview)
- [How It Works](#how-it-works)
- [Models](#models)
- [Dataset](#dataset)
- [Results](#results)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Training the Models](#training-the-models)
- [Running the Application](#running-the-application)
- [API](#api)
- [Important Notes](#important-notes)
- [Future Improvements](#future-improvements)

## Project Overview

Large and powerful models can achieve good image-classification performance, but deploying them on devices with limited compute and memory can be challenging.

EdgeMind investigates a teacher–student approach:

1. Train a Vision Transformer on CIFAR-10.
2. Train a MobileNetV3-Small student as a baseline.
3. Train another MobileNetV3-Small student using knowledge distillation from the teacher.
4. Compare the teacher, baseline student, and distilled student.
5. Serve the distilled student through a Flask API and a React frontend.

The goal is to transfer useful knowledge from the teacher to a smaller student model.

## How It Works

```text
                    CIFAR-10 Dataset
                           |
                 Training / Validation
                           |
              +------------+------------+
              |                         |
       Vision Transformer         MobileNetV3-Small
          (Teacher)                (Baseline Student)
              |                         |
        Teacher outputs          Supervised training
              |
              +----------+
                         |
                Knowledge Distillation
                         |
                 MobileNetV3-Small
                (Distilled Student)
                         |
                  Saved checkpoint
                         |
                    Flask API
                         |
                  React Frontend
                         |
                 Image Prediction
```

During distillation, the student learns from both:

- **Ground-truth labels**, using Cross-Entropy loss.
- **Teacher predictions**, using KL-Divergence loss.

The combined loss is:

\[
L = (1-\alpha)L_{CE} + \alpha T^2 L_{KL}
\]

Where:

- \(L_{CE}\) is the Cross-Entropy loss.
- \(L_{KL}\) is the KL-Divergence loss between teacher and student predictions.
- \(T\) is the distillation temperature.
- \(\alpha\) controls the contribution of the distillation loss.

The current training configuration uses **temperature = 4.0** and **alpha = 0.5**.

## Models

### Teacher: Vision Transformer

The teacher is a custom Vision Transformer configured for CIFAR-10:

- Image size: 32 × 32
- Patch size: 4 × 4
- Embedding dimension: 192
- Transformer depth: 6
- Attention heads: 3
- Output classes: 10

### Student: MobileNetV3-Small

The student uses the MobileNetV3-Small architecture from Torchvision, with its classifier adjusted for the 10 CIFAR-10 classes.

Two student versions are trained:

- **Baseline student:** trained using the ground-truth labels.
- **Distilled student:** trained using ground-truth labels and teacher predictions.

## Dataset

EdgeMind uses the **CIFAR-10** image-classification dataset.

- Number of classes: 10
- Training samples: 45,000
- Validation samples: 5,000
- Test samples: 10,000
- Image dimensions: 32 × 32 pixels
- Batch size: 64

The training data is split into training and validation sets using a fixed random seed. The same CIFAR-10 normalization values are used during training and inference.

The ten classes are:

`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, and `truck`.

## Results

The following are the recorded test-set results for the three models:

| Metric | Teacher (ViT) | Baseline Student | Distilled Student |
|---|---:|---:|---:|
| Test accuracy | 65.57% | 55.06% | 63.33% |
| Parameters | 1,806,538 | 1,528,106 | 1,528,106 |
| Model weights size | 6.89 MB | 5.88 MB | 5.88 MB |
| Checkpoint size | 20.77 MB | 17.73 MB | 17.74 MB |
| CPU inference time | 6.716 ms | 5.862 ms | 6.268 ms |
| GPU inference time | 4.186 ms | 7.641 ms | 8.008 ms |

### Key Observations

- The distilled student improves test accuracy by **8.27 percentage points** compared with the baseline student.
- The distilled student has **1,528,106 parameters**, compared with **1,806,538** in the teacher.
- The recorded distilled-student accuracy is **63.33%**, compared with **65.57%** for the teacher.
- In the recorded timing measurements, the student models were not faster than the teacher on GPU. Inference speed depends on the hardware and measurement setup, so the results should not be interpreted as a universal speed improvement.

The frontend also includes model comparison and evaluation views, including class-level metrics and a confusion matrix.

## Technology Stack

### Machine Learning

- Python
- PyTorch
- Torchvision
- CIFAR-10

### Backend

- Flask
- Flask-CORS
- Pillow

### Frontend

- React
- Vite
- JavaScript
- CSS

## Project Structure

```text
EdgeMind/
├── src/
│   ├── dataset.py
│   ├── teacher.py
│   ├── student.py
│   ├── distillation.py
│   ├── train_teacher.py
│   ├── train_student.py
│   └── train_distilled.py
│
├── deployment/
│   ├── app.py
│   └── frontend/
│       ├── public/
│       ├── src/
│       │   ├── components/
│       │   │   ├── Header.jsx
│       │   │   ├── ImageClassifier.jsx
│       │   │   ├── ModelComparison.jsx
│       │   │   ├── ModelEvaluation.jsx
│       │   │   ├── Overview.jsx
│       │   │   └── Sidebar.jsx
│       │   └── App.jsx
│       └── package.json
│
├── models/
│   ├── teacher/
│   └── student/
│
├── data/
├── results/
├── README.md
└── requirements.txt
```

The `models/`, `data/`, and `results/` directories shown above represent the expected locations for model checkpoints, dataset files, and evaluation outputs. Some of these files may be excluded from Git by `.gitignore`.

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/sachinkumar7106/EdgeMind.git
cd EdgeMind
```

### 2. Create and activate a Python virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

Install PyTorch and Torchvision using the command appropriate for your operating system and CPU/CUDA setup. Then install the backend dependencies:

```bash
pip install flask flask-cors pillow
```

The repository's `requirements.txt` should be populated with the exact project dependencies and compatible versions before using it as the single-command installation source.

### 4. Install frontend dependencies

```bash
cd deployment/frontend
npm install
```

## Training the Models

Run the training scripts from the repository root so that relative paths such as `./data` resolve as expected.

### Train the teacher

```bash
python -m src.train_teacher
```

### Train the baseline student

```bash
python -m src.train_student
```

### Train the distilled student

The distilled-student training script expects the trained teacher checkpoint to be available at the path configured in the script.

```bash
python -m src.train_distilled
```

Check the checkpoint paths in the training scripts before running them. Training scripts may download CIFAR-10 if it is not already present locally and the environment has network access.

## Running the Application

The application uses a Flask backend and a Vite development server for the React frontend. Start them in separate terminals.

### 1. Start the Flask backend

From the repository root:

```bash
python deployment/app.py
```

The backend runs at:

```text
http://127.0.0.1:5000
```

The deployed model checkpoint is expected at:

```text
deployment/models/student/best_distilled_student.pth
```

Make sure the trained distilled-student checkpoint is placed at the path configured by `MODEL_PATH` in `deployment/app.py`. If your checkpoint is stored elsewhere, update that path.

### 2. Start the React frontend

In a second terminal:

```bash
cd deployment/frontend
npm install
npm run dev
```

Open the local URL printed by Vite, usually:

```text
http://localhost:5173
```

Use the Image Classification page to upload an image and view the predicted CIFAR-10 class and class probabilities.

## API

The Flask backend exposes the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Checks whether the API is running and the model is loaded. |
| `POST` | `/api/predict` | Accepts an image and returns the predicted class and probabilities. |

### Prediction request

Send a `multipart/form-data` request to `/api/predict` with the image in a field named `image`.

Example using `curl`:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -F "image=@path/to/image.jpg"
```

The prediction response includes the predicted class, confidence, and probabilities for all ten CIFAR-10 classes.

## Important Notes

- The inference API is configured for CIFAR-10 classes and 32 × 32 input images. It resizes uploaded images before applying the CIFAR-10 normalization.
- The frontend currently sends requests to `http://127.0.0.1:5000`. This is suitable for local development; update the API URL configuration when deploying the frontend and backend to hosted environments.
- Model checkpoint files are excluded by the repository's `.gitignore` rules. You must provide the required checkpoint separately when setting up the application.
- The backend and frontend must both be running for local image classification to work.
- The reported timing values are measurements from a particular environment and should be re-measured on the target deployment hardware.

## Future Improvements

Potential next steps for the project include:

- Add quantization and pruning to further reduce model size and inference cost.
- Benchmark latency, memory usage, and energy consumption on resource-constrained hardware.
- Evaluate the models on additional datasets and input conditions.
- Add configurable backend URLs for development and production.
- Add backend-side upload-size validation and stronger input validation.
- Document reproducible environment versions and checkpoint download/setup instructions.

## Author

**Sachin Kumar**

GitHub: [sachinkumar7106](https://github.com/sachinkumar7106)

## License

Add the license you intend to use for this project before distributing or reusing the code.
