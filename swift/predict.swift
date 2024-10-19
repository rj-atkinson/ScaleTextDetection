import Foundation
import CoreML
import Vision
import CoreVideo
import AppKit

struct ScaleTextDetector {
    let model: VNCoreMLModel?

    init(modelURL: URL) {
        self.model = Self.loadModel(modelURL: modelURL)
    }

    // Function to load an image from the file system and convert it to a CVPixelBuffer
    public static func imageToCVPixelBuffer(imagePath: String) -> CVPixelBuffer? {
        guard let nsImage = NSImage(contentsOfFile: imagePath),
            let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            print("Error: Unable to load image or convert to CGImage.")
            return nil
        }
        
        let frameSize = CGSize(width: cgImage.width, height: cgImage.height)
        
        var pixelBuffer: CVPixelBuffer?
        let pixelBufferAttributes: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ]
        
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                        Int(frameSize.width),
                                        Int(frameSize.height),
                                        kCVPixelFormatType_32ARGB,  // YOLO typically works with RGB or ARGB images
                                        pixelBufferAttributes as CFDictionary,
                                        &pixelBuffer)
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            print("Error: Unable to create CVPixelBuffer.")
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        
        // Create a Core Graphics context to draw the CGImage into the CVPixelBuffer
        let pixelBufferBaseAddress = CVPixelBufferGetBaseAddress(buffer)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: pixelBufferBaseAddress,
                                    width: Int(frameSize.width),
                                    height: Int(frameSize.height),
                                    bitsPerComponent: 8,
                                    bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                    space: colorSpace,
                                    bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
            print("Error: Unable to create CGContext.")
            CVPixelBufferUnlockBaseAddress(buffer, [])
            return nil
        }
        
        // Draw the image into the pixel buffer context
        context.draw(cgImage, in: CGRect(origin: .zero, size: frameSize))
        
        CVPixelBufferUnlockBaseAddress(buffer, [])
        
        return buffer
    }

    // Function to load the CoreML model
    public static func loadModel(modelURL: URL) -> VNCoreMLModel? {
        do {
            let compiledModelURL = try MLModel.compileModel(at: modelURL)
            let model = try MLModel(contentsOf: compiledModelURL)
            return try VNCoreMLModel(for: model)
        } catch {
            print("Failed to load model: \(error)")
            return nil
        }
    }

    public func runYOLOModel(pixelBuffer: CVPixelBuffer) {
        guard let model else {
            print("model not defined")
            return
        }

        Self.runYOLOModel(model: model, pixelBuffer: pixelBuffer)
    }

    // Function to run the image (CVPixelBuffer) through the model
    public static func runYOLOModel(model: VNCoreMLModel, pixelBuffer: CVPixelBuffer) {
        let request = VNCoreMLRequest(model: model) { (request, error) in
            if let results = request.results as? [VNRecognizedObjectObservation] {
                for observation in results {
                    print("Detected object: \(observation.labels.first?.identifier ?? "Unknown")")
                    print("Confidence: \(observation.labels.first?.confidence ?? 0)")
                    print("Bounding Box: \(observation.boundingBox)")
                    
                    // For converting bounding box coordinates to pixel values:
                    let bb = observation.boundingBox
                    let width = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
                    let height = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
                    print("YOLO Coords: \(bb.minX * width), \(bb.minY * height), \(bb.maxX * width), \(bb.maxY * height)")
                }
            } else {
                print("No objects detected or invalid results.")
            }
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        do {
            try handler.perform([request])
        } catch {
            print("Failed to perform request: \(error)")
        }
    }
}

// Check if we have enough command-line arguments
guard CommandLine.argc == 3 else {
    print("Usage: swift run_yolo.swift <path_to_model.mlpackage> <path_to_image>")
    exit(1)
}

// Parse the arguments
let modelPath = CommandLine.arguments[1]
let imagePath = CommandLine.arguments[2]

// Load the model
let modelURL: URL = .init(fileURLWithPath: modelPath)
let scaleTextDetector: ScaleTextDetector = .init(modelURL: modelURL)

guard scaleTextDetector.model != nil else {
    print("Error loading model.")
    exit(1)
}

// Convert the image to CVPixelBuffer
guard let pixelBuffer = ScaleTextDetector.imageToCVPixelBuffer(imagePath: imagePath) else {
    print("Error loading image or creating CVPixelBuffer.")
    exit(1)
}

// Run the YOLO model on the CVPixelBuffer
scaleTextDetector.runYOLOModel(pixelBuffer: pixelBuffer)
