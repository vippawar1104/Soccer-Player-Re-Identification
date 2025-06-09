# ⚽ Soccer Player Re-Identification – YOLOv11 + Custom Tracker

This project performs **player re-identification** in a soccer match video using **YOLOv11** for detection and a custom feature-based tracker. The tracker assigns **consistent player IDs**, even when players leave and re-enter the frame.

---

## 🎥 Demo Output

🔗 [Watch the Output Video](https://drive.google.com/file/d/XXXX/view)  
*(Replace this link with your actual Google Drive link)*

---

## 🧠 Features

- ✅ Player detection using YOLOv11
- 🔁 Consistent tracking using visual + spatial features
- 🎯 Handles re-identification of players leaving and re-entering
- 📊 Exports full tracking data in JSON format
- 🧩 Pure Python (no external trackers like DeepSORT)

---

## 📦 Repository Structure
soccer-reid/
├── main.py # Full code with tracking logic
├── best.pt # YOLOv11 model (optional: link via Drive)
├── 15sec_input_720p.mp4 # Input video (optional: link via Drive)
├── output_with_tracking.mp4 # Output video (host via Drive)
├── tracking_results.json # JSON of player detections and tracks
├── requirements.txt # Python dependencies
└── README.md # You're reading it!

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/vippawar1104/soccer-reid.git
cd soccer-reid
```

### 2. Create Virtual Environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install ultralytics opencv-python-headless numpy matplotlib torch torchvision scipy scikit-learn
```

---

## ▶️ How to Run

### Local Script (main.py)
1. Place `best.pt` and `15sec_input_720p.mp4` in the project folder.
2. Run:
```bash
python main.py
```

### Google Colab (Recommended for Easy Testing)
1. Open the Colab notebook (upload main.py content).
2. Upload `best.pt` and `15sec_input_720p.mp4` when prompted.
3. Output will auto-download (`output_with_tracking.mp4` + `tracking_results.json`).

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `output_with_tracking.mp4` | Video with annotated player IDs |
| `tracking_results.json` | Frame-by-frame re-ID tracking data |

---

## 📘 How It Works

- **Detection**: YOLOv11 detects players in each frame.
- **Feature Extraction**: Color histograms, texture, position features are computed.
- **Tracking**: Matches players across frames based on feature similarity + spatial proximity.
- **Re-Identification**: If a player disappears and returns, the tracker re-assigns the same ID based on appearance.

---

## 🧪 Evaluation Criteria Addressed

| Criteria | Implementation |
|----------|----------------|
| ✔️ Accuracy & Re-ID Logic | Feature-based + IoU + distance matching |
| ✔️ Simplicity & Modularity | Clean, modular code with docstrings |
| ✔️ Documentation | This README + in-code comments |
| ✔️ Runtime Efficiency | Frame-wise optimized tracking loop |
| ✔️ Creativity | Custom tracker — no 3rd-party dependencies |

---

## 🔗 Resources

- 📦 YOLOv11 Model: [Upload your best.pt to Drive and link here]
- 🎥 Input Video: [Upload 15sec_input_720p.mp4 and link here]
- 🎬 Output Video: [https://drive.google.com/file/d/XXXX/view]

---

## 🙋 Author

**Vipul Pawar**  
📧 vipulpawar81077@gmail.com  
🎓 B.Tech (ECE), IIIT Nagpur  
🔗 [GitHub](https://github.com/vippawar1104) | [LinkedIn](https://linkedin.com/in/your-profile)

---

### 📌 Final To-Do (Your Side)

- [ ] Upload `output_with_tracking.mp4` to Google Drive  
- [ ] Replace `https://drive.google.com/file/d/XXXX/view` in the README
- [ ] Push all files (except big `.mp4`) to GitHub

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📚 References

- [YOLOv11 Paper](https://arxiv.org/abs/XXXX.XXXXX)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Ultralytics YOLO](https://github.com/ultralytics/yolov5)
