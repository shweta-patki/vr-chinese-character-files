using System;
using System.IO;
using System.Linq;
using Tesseract;

internal static class Program
{
    private static int Main(string[] args)
    {
        // Usage: dotnet run -- [datasetDir] [lang] [tessdataPath]
        // Defaults: datasetDir = "dataset", lang = "chi_sim", tessdataPath = ./tessdata
        string datasetDir = args.Length > 0 ? args[0] : Path.Combine(AppContext.BaseDirectory, "dataset");
        string lang = args.Length > 1 ? args[1] : "chi_sim";
        string tessdata = args.Length > 2 ? args[2] : Path.Combine(AppContext.BaseDirectory, "tessdata");

        Console.WriteLine($"Dataset: {datasetDir}");
        Console.WriteLine($"Language: {lang}");
        Console.WriteLine($"Tessdata: {tessdata}");

        if (!Directory.Exists(datasetDir))
        {
            Console.Error.WriteLine($"Dataset directory not found: {datasetDir}");
            return 2;
        }

        if (!Directory.Exists(tessdata))
        {
            Console.Error.WriteLine($"Tessdata directory not found: {tessdata}");
            Console.Error.WriteLine("Download the appropriate traineddata files (for example chi_sim.traineddata) and place them into the tessdata folder.");
            return 3;
        }

        try
        {
            using var engine = new TesseractEngine(tessdata, lang, EngineMode.Default);

            string[] imageExtensions = new[] { ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff" };

            // Process top-level dataset folders (train, validation, etc.) and any nested class folders.
            foreach (var top in Directory.EnumerateDirectories(datasetDir))
            {
                Console.WriteLine($"Processing folder: {top}");

                foreach (var classDir in Directory.EnumerateDirectories(top))
                {
                    Console.WriteLine($"  Class: {Path.GetFileName(classDir)}");

                    var files = Directory.EnumerateFiles(classDir)
                        .Where(f => imageExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
                        .ToList();

                    foreach (var file in files)
                    {
                        try
                        {
                            using var img = Pix.LoadFromFile(file);
                            using var page = engine.Process(img);
                            string text = page.GetText()?.Trim() ?? string.Empty;
                            float conf = page.GetMeanConfidence();
                            Console.WriteLine($"    {Path.GetFileName(file)} => '{text}' (conf {conf:F2})");

                            // Write the recognized text beside the image file for later inspection
                            File.WriteAllText(file + ".txt", text);
                        }
                        catch (Exception ex)
                        {
                            Console.Error.WriteLine($"    Error processing {file}: {ex.Message}");
                        }
                    }
                }
            }

            Console.WriteLine("OCR pass complete.");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Fatal error: {ex.Message}");
            return 1;
        }
    }
}
