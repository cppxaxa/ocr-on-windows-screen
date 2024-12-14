using OpenCvSharp;
using OpenCvSharp.Dnn;
using Tesseract;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Started");

        // Load the model and input image
        var net = CvDnn.ReadNet("frozen_east_text_detection.pb");
        var input = Cv2.ImRead("image.jpg");

        // Create a diagnostic image (copy of the input image)
        Mat diagnosticImage = input.Clone();

        // Save original image dimensions
        int originalWidth = input.Width;
        int originalHeight = input.Height;

        // Define padding for X and Y directions
        int paddingX = 5;
        int paddingY = 5;

        // Prepare the input blob (resized to 640x640)
        var blob = CvDnn.BlobFromImage(input, 1.0, new Size(640, 640), new Scalar(123.68, 116.78, 103.94), true, false);
        net.SetInput(blob);

        // Get network outputs
        Mat scores = net.Forward("feature_fusion/Conv_7/Sigmoid");
        Mat geometry = net.Forward("feature_fusion/concat_3");

        // Decode bounding boxes
        float confidenceThreshold = 0.3f;
        var boxes = Decode(scores, geometry, confidenceThreshold, originalWidth / 640.0f, originalHeight / 640.0f);

        // Merge bounding boxes
        var mergedBoxes = MergeBoundingBoxes(boxes, 10, 10);

        // Tesseract OCR
        bool runOcr = true;
        if (runOcr)
        {
            using var ocr = new TesseractEngine(@"./tessdata", "eng", EngineMode.Default);
            var textMap = new Dictionary<string, OpenCvSharp.Point>();

            foreach (var box in mergedBoxes)
            {
                int x1 = Math.Max(box.X - paddingX, 0);
                int y1 = Math.Max(box.Y - paddingY, 0);
                int x2 = Math.Min(box.X + box.Width + paddingX, input.Width);
                int y2 = Math.Min(box.Y + box.Height + paddingY, input.Height);

                var cropped = new Mat(input, new OpenCvSharp.Rect(x1, y1, x2 - x1, y2 - y1));
                Cv2.Rectangle(diagnosticImage, box, Scalar.Red, 2);

                Pix croppedPix = MatToPix(cropped);
                using (var page = ocr.Process(croppedPix))
                {
                    var text = page.GetText().Trim();
                    if (!string.IsNullOrEmpty(text))
                    {
                        Cv2.PutText(diagnosticImage, text, new OpenCvSharp.Point(box.X, box.Y - 5), HersheyFonts.HersheySimplex, 0.8, Scalar.BlueViolet, 2);
                        textMap[text] = new OpenCvSharp.Point(box.X + box.Width / 2, box.Y + box.Height / 2);
                    }
                }
            }

            Console.WriteLine("Detected Text and Center Coordinates:");
            foreach (var entry in textMap)
            {
                Console.WriteLine($"Text: '{entry.Key}' at Center: {entry.Value}");
            }
        }
        else
        {
            // Only draw bounding boxes and skip OCR
            foreach (var box in mergedBoxes)
            {
                Cv2.Rectangle(diagnosticImage, box, Scalar.Red, 2);  // Draw red bounding boxes
            }
        }

        Gtk.Application.Init();

        // Get screen size (viewport size)
        int screenWidth = (int)(Gdk.Screen.Default.Width * 0.9);
        int screenHeight = (int)(Gdk.Screen.Default.Height * 0.9);

        // Resize the image to fit the screen resolution
        Mat resizedImage = new Mat();
        Cv2.Resize(diagnosticImage, resizedImage, new OpenCvSharp.Size(screenWidth, screenHeight));

        // Show the resized image
        Cv2.ImShow("Detected Text", resizedImage);
        Cv2.WaitKey(0);
    }

    static List<OpenCvSharp.Rect> Decode(Mat scores, Mat geometry, float scoreThreshold, float scaleX, float scaleY)
    {
        var height = scores.Size(2);
        var width = scores.Size(3);
        var boxes = new List<OpenCvSharp.Rect>();

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float score = scores.At<float>(0, 0, y, x);
                if (score >= scoreThreshold)
                {
                    float offsetX = x * 4.0f;
                    float offsetY = y * 4.0f;

                    float angle = geometry.At<float>(0, 4, y, x);
                    float cosA = (float)Math.Cos(angle);
                    float sinA = (float)Math.Sin(angle);

                    float h = geometry.At<float>(0, 0, y, x) + geometry.At<float>(0, 2, y, x);
                    float w = geometry.At<float>(0, 1, y, x) + geometry.At<float>(0, 3, y, x);

                    float endX = offsetX + (cosA * geometry.At<float>(0, 1, y, x)) + (sinA * geometry.At<float>(0, 2, y, x));
                    float endY = offsetY - (sinA * geometry.At<float>(0, 1, y, x)) + (cosA * geometry.At<float>(0, 2, y, x));

                    float startX = endX - w;
                    float startY = endY - h;

                    var rect = new OpenCvSharp.Rect(
                        (int)(startX * scaleX),
                        (int)(startY * scaleY),
                        (int)(w * scaleX),
                        (int)(h * scaleY)
                    );

                    boxes.Add(rect);
                }
            }
        }
        return boxes;
    }



    static List<OpenCvSharp.Rect> MergeBoundingBoxes(List<OpenCvSharp.Rect> boxes, int mergeThresholdX, int mergeThresholdY)
    {
        return RectMerger.MergeOverlappingRects(boxes, mergeThresholdX, mergeThresholdY);
    }

    static Pix MatToPix(Mat mat)
    {
        Cv2.ImEncode(".png", mat, out byte[] bytes);
        return Pix.LoadFromMemory(bytes);
    }
}

class RectMerger
{
    public static List<OpenCvSharp.Rect> MergeOverlappingRects(List<OpenCvSharp.Rect> boxes, int mergeThresholdX = 0, int mergeThresholdY = 0)
    {
        if (boxes == null || boxes.Count == 0)
            return new List<OpenCvSharp.Rect>();

        List<OpenCvSharp.Rect> mergedRects = new List<OpenCvSharp.Rect>();
        bool[] merged = new bool[boxes.Count];

        for (int i = 0; i < boxes.Count; i++)
        {
            if (merged[i]) continue;
            OpenCvSharp.Rect current = boxes[i];

            for (int j = i + 1; j < boxes.Count; j++)
            {
                if (merged[j]) continue;

                if (current.IntersectsWith(boxes[j], mergeThresholdX, mergeThresholdY))
                {
                    current = MergeTwoRects(current, boxes[j]);
                    merged[j] = true;
                }
            }
            mergedRects.Add(current);
        }
        return mergedRects;
    }

    private static OpenCvSharp.Rect MergeTwoRects(OpenCvSharp.Rect a, OpenCvSharp.Rect b)
    {
        int x = Math.Min(a.Left, b.Left);
        int y = Math.Min(a.Top, b.Top);
        int right = Math.Max(a.Right, b.Right);
        int bottom = Math.Max(a.Bottom, b.Bottom);
        return new OpenCvSharp.Rect(x, y, right - x, bottom - y);
    }
}

// Extension method for checking intersection
public static class RectExtensions
{
    public static bool IntersectsWith(this OpenCvSharp.Rect a, OpenCvSharp.Rect b, int thresholdX, int thresholdY)
    {
        return a.Left - thresholdX < b.Right && a.Right + thresholdX > b.Left &&
               a.Top - thresholdY < b.Bottom && a.Bottom + thresholdY > b.Top;
    }
}
