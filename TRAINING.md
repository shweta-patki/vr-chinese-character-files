**Training OCR / Classifier locally**

This document explains how to train a simple character classifier on your PC using the dataset in `dataset/train` and `dataset/validation` and export it for use in Unity/Quest.

1) Requirements

- Python 3.8+
- pip packages: `tensorflow`, `tf2onnx` (optional), `onnx` (optional)

Install with:

```powershell
python -m pip install --upgrade pip
pip install tensorflow tf2onnx onnx
```

2) Training

- Script: `training/train_classifier.py`
- Basic usage (from `ChineseOcr` project folder):

```powershell
python training/train_classifier.py --data_dir dataset --out_dir models --img_size 128 --epochs 30
```

- The script expects `dataset/train/<class>` and `dataset/validation/<class>` folders of images.
- It will save:
  - `models/saved_model/` (TensorFlow SavedModel)
  - `models/model_final.h5` (Keras H5)
  - `models/classes.txt` (list of class names in order)
  - `models/model.onnx` (if tf2onnx is available)
  - `models/model.tflite` (if TFLite conversion succeeds)

3) Export for Unity / Quest

- Recommended for Unity: export to ONNX (`models/model.onnx`) and use Unity Barracuda or ONNX Runtime plugin. If ONNX export fails, you can use the SavedModel and convert externally.
- Alternatively export `model.tflite` and use a TFLite runtime plugin for Unity or write an Android plugin that uses TFLite.
- Keep `models/classes.txt` alongside the model and include it in Unity to map numeric predictions back to characters.

4) Notes

- If your images are single characters (one character per image) a classifier is simpler and faster than full Tesseract OCR for single-character recognition.
- If your dataset contains multi-character text or variable layouts, consider training a detection+recognition pipeline (PaddleOCR or CRNN+CTC).
