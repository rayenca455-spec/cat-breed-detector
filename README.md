# 🐾 Pet Breed Detector

An AI-powered web app that detects the breed of your cat or dog from a photo.
Upload an image and get the **top 3 breed predictions** with confidence scores.

🚀 **[Try it live on Hugging Face](https://huggingface.co/spaces/RAYY-1920/pet-breed-detector)**

---

## Demo

> Upload a photo → Get instant breed predictions

Supports **12 cat breeds** and **25 dog breeds**.

---

## How it works

- Fine-tuned **ResNet18** (PyTorch) on the Oxford-IIIT Pet Dataset
- Returns top 3 predictions with confidence percentages
- REST API built with **FastAPI**
- Interactive web interface built with **Gradio**

---

## Supported Breeds

**Cats:** Abyssinian, Bengal, Birman, Bombay, British Shorthair, Egyptian Mau, Maine Coon, Persian, Ragdoll, Russian Blue, Siamese, Sphynx

**Dogs:** American Bulldog, American Pit Bull Terrier, Basset Hound, Beagle, Boxer, Chihuahua, English Cocker Spaniel, English Setter, German Shorthaired, Great Pyrenees, Havanese, Japanese Chin, Keeshond, Leonberger, Miniature Pinscher, Newfoundland, Pomeranian, Pug, Saint Bernard, Samoyed, Scottish Terrier, Shiba Inu, Staffordshire Bull Terrier, Wheaten Terrier, Yorkshire Terrier

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| PyTorch | Model training & inference |
| ResNet18 | CNN architecture |
| FastAPI | REST API |
| Gradio | Web interface |
| Hugging Face Spaces | Deployment |

---

## Project Structure

```
├── app.py              # Gradio web interface
├── main.py             # FastAPI REST API
├── breeds.json         # List of supported breeds
├── requirements.txt    # Dependencies
└── cat_breed_model.pth # Trained model (hosted on Hugging Face)
```

---

## Run locally

```bash
git clone https://github.com/rayenca455-spec/cat-breed-detector
cd cat-breed-detector
pip install -r requirements.txt
python app.py
```

> The model file `cat_breed_model.pth` is hosted on Hugging Face due to its size.
> Download it from the [Space files](https://huggingface.co/spaces/RAYY-1920/pet-breed-detector/tree/main) and place it in the root folder.

---

## Author

**Rayen Helali** — Big Data Student | Machine Learning & Deep Learning

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Rayen_Helali-blue?logo=linkedin)](https://www.linkedin.com/in/helali-rayen-5a929641a/)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Live_Demo-yellow)](https://huggingface.co/spaces/RAYY-1920/pet-breed-detector)
