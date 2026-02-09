# Chinese OCR

This small C# program runs OCR on the images in the `dataset` folder using Tesseract.

Requirements
- .NET 8 SDK (the project targets `net8.0`).
- Windows (this uses `Tesseract` and `System.Drawing.Common`), or ensure native Tesseract / Leptonica libs are available on other platforms.
- Download appropriate `tessdata` traineddata files for the Chinese language (for example `chi_sim.traineddata`) and place them into a `tessdata` folder next to the built executable (or provide the path as an argument).

Quick setup

1. From the project directory run:

```powershell
dotnet restore
dotnet build
```

2. Install/get the Chinese traineddata file(s):

- Recommended: download `chi_sim.traineddata` from the official tesseract tessdata repo, e.g. https://github.com/tesseract-ocr/tessdata or https://github.com/tesseract-ocr/tessdata_best depending on accuracy/size.
- Create a `tessdata` folder next to the project output (or in the project root while running via `dotnet run`).

3. Place the traineddata file(s) in the `tessdata` folder.

Running

Default (expects `dataset` in project output directory and `tessdata` in app folder):

```powershell
dotnet run --project . -- "dataset" "chi_sim" "./tessdata"
```

Arguments
- `datasetDir` (optional): path to the dataset folder. Default `dataset`.
- `lang` (optional): Tesseract language code. Default `chi_sim`.
- `tessdataPath` (optional): path to the `tessdata` folder containing `.traineddata` files.

Output
- The program logs recognized text and writes a `.txt` file alongside each image containing the recognized text.

Notes
- If you encounter native library errors when loading Tesseract, you may need to install the Tesseract runtime or place the native DLLs for Leptonica/Tesseract on your PATH.
- For large datasets consider batching or running processing in parallel; this sample is sequential for clarity and simplicity.
