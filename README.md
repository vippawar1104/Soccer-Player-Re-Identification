# ⚽ Soccer Player Re-Identification – YOLOv11 + Custom Tracker

This project performs **player re-identification** in a soccer match video using **YOLOv11** for detection and a custom feature-based tracker. The tracker assigns **consistent player IDs**, even when players leave and re-enter the frame.

---

## 🎥 Demo Output

🔗 [Watch the Output Video](https://drive.google.com/file/d/15XSbG_dbblt5shyntvHEs_-CIxnqQmjw/view?usp=drive_link)  

---

## 🧠 Features

- ✅ Player detection using YOLOv11
- 🔁 Consistent tracking using visual + spatial features
- 🎯 Handles re-identification of players leaving and re-entering
- 📊 Exports full tracking data in JSON format
- 🧩 Pure Python (no external trackers like DeepSORT)

---

## 📦 Repository Structure

```
soccer-reid/
├── main.py                   # Full code with tracking logic
├── best.pt                   # YOLOv11 model (optional: link via Drive)
├── 15sec_input_720p.mp4      # Input video (optional: link via Drive)
├── output_with_tracking.mp4  # Output video (host via Drive)
├── tracking_results.json     # JSON of player detections and tracks
├── requirements.txt          # Python dependencies
└── README.md                 # You're reading it!
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone [https://github.com/vippawar1104/Soccer-Player-Re-Identification]
cd soccer-reid
```

### 2. Create a Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Or install manually:**

```bash
pip install ultralytics opencv-python-headless numpy matplotlib torch torchvision scipy scikit-learn
```

---

## ▶️ How to Run

### Option 1: Local Script (`main.py`)

1. Place `best.pt` and `15sec_input_720p.mp4` in the project folder.
2. Run:

```bash
python main.py
```

### Option 2: Google Colab (Recommended for Easy Testing)

1. Open a Colab notebook and paste the code from `main.py`.
2. Upload `best.pt` and `15sec_input_720p.mp4` when prompted.
3. Output files (`output_with_tracking.mp4`, `tracking_results.json`) will download automatically.

---

## 📊 Output Files

| File                      | Description                                |
|---------------------------|--------------------------------------------|
| `output_with_tracking.mp4` | Annotated video with consistent player IDs |
| `tracking_results.json`   | Frame-wise re-ID tracking results          |

---

## 📘 How It Works

- **Detection**: YOLOv11 detects players in each frame.
- **Feature Extraction**: Color histograms, texture, and positional features.
- **Tracking**: Players are matched frame-to-frame using similarity + IoU.
- **Re-Identification**: Reappearance is handled by similarity scoring and ID assignment.

---

## 🧪 Evaluation Criteria Addressed

| Criteria                    | Implementation Highlights                       |
|-----------------------------|--------------------------------------------------|
| ✔️ Accuracy & Re-ID Logic   | Feature-based + IoU + distance matching          |
| ✔️ Simplicity & Modularity  | Clean code with reusable functions/classes       |
| ✔️ Documentation            | Inline comments + this detailed README           |
| ✔️ Runtime Efficiency       | Frame-optimized processing using NumPy           |
| ✔️ Creativity               | Custom-built re-ID tracker (no external libs)    |

---

## 🔗 Resources

- 📦 YOLOv11 Model: [best.pt](https://drive.google.com/file/d/1ZqyOaksKkVyzurosrP1DJa2M6E2rmBnn/view?usp=drive_link)
- 🎥 Input Video: [15sec_input_720p.mp4](https://drive.google.com/file/d/1GYWks6NURX2nMBOH2RCNT1NKELHN2rZo/view?usp=drive_link)
- 🎬 Output Video: [Output with tracking](https://drive.google.com/file/d/15XSbG_dbblt5shyntvHEs_-CIxnqQmjw/view?usp=drive_link)

---

## 🙋 Author

**Vipul Pawar**  
📧 vipulpawar81077@gmail.com  
🎓 B.Tech (ECE), IIIT Nagpur  
🔗 [GitHub](https://github.com/vippawar1104)  
🔗 [LinkedIn](https://www.linkedin.com/in/vipul-pawar-1104vip)


---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and open a pull request.

---

## 📚 References

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Re-Identification Concepts](https://arxiv.org/abs/1801.10352)
