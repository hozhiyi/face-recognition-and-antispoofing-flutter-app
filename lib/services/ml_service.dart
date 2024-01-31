import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:face_net_authentication/pages/db/databse_helper.dart';
import 'package:face_net_authentication/pages/models/user.model.dart';
import 'package:face_net_authentication/services/image_converter.dart';
import 'package:flutter/services.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'package:image/image.dart' as imglib;
import 'package:onnxruntime/onnxruntime.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

Future<OrtSession> loadModelFromAssets(String assetPath) async {
  OrtEnv.instance.init();
  final sessionOptions = OrtSessionOptions();
  const assetFileName = 'assets/FaceBagNet_color_96.onnx';
  final rawAssetFile = await rootBundle.load(assetFileName);
  final bytes = rawAssetFile.buffer.asUint8List();
  final session = OrtSession.fromBuffer(bytes, sessionOptions);

  // // Get the directory to store the model file
  // final dir = await getApplicationDocumentsDirectory();

  // // Create a file in the directory
  // final file = File('${dir.path}/FaceBagNet_color_96.onnx');

  // // Write the model file
  // final modelData = await rootBundle.load(assetPath);
  // await file.writeAsBytes(
  //   modelData.buffer
  //       .asUint8List(modelData.offsetInBytes, modelData.lengthInBytes),
  // );

  // // Create session options
  // final sessionOptions = OrtSessionOptions();

  // // Load the model
  // final session = OrtSession.fromFile(file, sessionOptions);

  return session;
}

class MLService {
  Interpreter? _interpreter;
  double threshold = 0.5;

  // Interpreter? _antiSpoofingInterpreter;
  double antiSpoofingThreshold = 0.5;

  List _predictedData = [];
  List get predictedData => _predictedData;
  double dist = 0;

  Future initialize() async {
    late Delegate delegate;
    try {
      if (Platform.isAndroid) {
        delegate = GpuDelegateV2(
          options: GpuDelegateOptionsV2(isPrecisionLossAllowed: false),
        );
      } else if (Platform.isIOS) {
        delegate = GpuDelegate(
          options: GpuDelegateOptions(
            allowPrecisionLoss: true,
          ),
        );
      }
      var interpreterOptions = InterpreterOptions()..addDelegate(delegate);

      this._interpreter = await Interpreter.fromAsset(
          'assets/ep050-loss23.614.tflite', //mobilefacenet.tflite
          options: interpreterOptions);
    } catch (e) {
      print('Failed to load model.');
      print(e);
    }
  }

  void setCurrentPrediction(CameraImage cameraImage, Face? face) {
    if (_interpreter == null) throw Exception('Interpreter is null');
    if (face == null) throw Exception('Face is null');

    // List input = _preProcess(cameraImage, face);
    // _preProcess starts
    imglib.Image croppedImage = _cropFace(cameraImage, face);
    imglib.Image img;
    img = imglib.copyResizeCropSquare(croppedImage, 112);
    List input = imageToByteListFloat32(img);
    // _preProcess ends

    input = input.reshape([1, 112, 112, 3]);
    List output = List.generate(1, (index) => List.filled(256, 0));

    this._interpreter?.run(input, output);

    // print("==> ori output : " + output.toString());

    output = output.reshape([256]);

    // print("==> mod output : " + output.toString());

    this._predictedData = List.from(output);
  }

  Future<User?> predict() async {
    // CHECK
    // print("===> this._predictedData" + this._predictedData.toString());
    return _searchResult(this._predictedData);
  }

  List _preProcess(CameraImage image, Face faceDetected) {
    imglib.Image croppedImage = _cropFace(image, faceDetected);
    imglib.Image img;

    img = imglib.copyResizeCropSquare(croppedImage, 96);

    Float32List imageAsList = imageToByteListFloat32Spoof(img);

    // normalization
    for (int i = 0; i < imageAsList.length; i++) {
      imageAsList[i] = imageAsList[i];
    }
    return imageAsList;
  }

  imglib.Image _cropFace(CameraImage image, Face faceDetected) {
    imglib.Image convertedImage = _convertCameraImage(image);
    double x = faceDetected.boundingBox.left - 10.0;
    double y = faceDetected.boundingBox.top - 10.0;
    double w = faceDetected.boundingBox.width + 10.0;
    double h = faceDetected.boundingBox.height + 10.0;
    return imglib.copyCrop(
        convertedImage, x.round(), y.round(), w.round(), h.round());
  }

  imglib.Image _convertCameraImage(CameraImage image) {
    var img = convertToImage(image);
    var img1 = imglib.copyRotate(img, -90);
    return img1;
  }

  Float32List imageToByteListFloat32(imglib.Image image) {
    var convertedBytes = Float32List(1 * 112 * 112 * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;

    for (var i = 0; i < 112; i++) {
      for (var j = 0; j < 112; j++) {
        var pixel = image.getPixel(j, i);
        // buffer[pixelIndex++] = (imglib.getRed(pixel) - 128) / 128;
        // buffer[pixelIndex++] = (imglib.getGreen(pixel) - 128) / 128;
        // buffer[pixelIndex++] = (imglib.getBlue(pixel) - 128) / 128;
        buffer[pixelIndex++] = imglib.getRed(pixel) / 255;
        buffer[pixelIndex++] = imglib.getGreen(pixel) / 255;
        buffer[pixelIndex++] = imglib.getBlue(pixel) / 255;
      }
    }
    return convertedBytes.buffer.asFloat32List();
  }

  Float32List imageToByteListFloat32Spoof(imglib.Image image) {
    var convertedBytes = Float32List(1 * 96 * 96 * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;

    for (var i = 0; i < 96; i++) {
      for (var j = 0; j < 96; j++) {
        var pixel = image.getPixel(j, i);
        // buffer[pixelIndex++] = (imglib.getRed(pixel) - 128) / 128;
        // buffer[pixelIndex++] = (imglib.getGreen(pixel) - 128) / 128;
        // buffer[pixelIndex++] = (imglib.getBlue(pixel) - 128) / 128;
        buffer[pixelIndex++] = imglib.getRed(pixel) / 255;
        buffer[pixelIndex++] = imglib.getGreen(pixel) / 255;
        buffer[pixelIndex++] = imglib.getBlue(pixel) / 255;
      }
    }
    return convertedBytes.buffer.asFloat32List();
  }

  Future<User?> _searchResult(List predictedData) async {
    DatabaseHelper _dbHelper = DatabaseHelper.instance;

    List<User> users = await _dbHelper.queryAllUsers();
    double minDist = 999;
    double currDist = 0.0;
    User? predictedResult;

    print('users.length=> ${users.length}');

    for (User u in users) {
      // CHECK
      // print("===> u.modelData" + (u.modelData).toString());
      // print("===> predictedData" + predictedData.toString());
      currDist = _euclideanDistance(u.modelData, predictedData);
      // currDist = _cosineDistance(u.modelData, predictedData);

      if (currDist <= threshold && currDist < minDist) {
        print('User ID: ${u.user}, Current Distance: $currDist');
        minDist = currDist;
        predictedResult = u;
      }
    }
    return predictedResult;
  }

  double _euclideanDistance(List? e1, List? e2) {
    if (e1 == null || e2 == null) throw Exception("Null argument");

    double sum = 0.0;
    for (int i = 0; i < e1.length; i++) {
      sum += pow((e1[i] - e2[i]), 2);
    }
    return sqrt(sum);
  }

  void setPredictedData(value) {
    this._predictedData = value;
  }

  double _cosineDistance(List? e1, List? e2) {
    try {
      if (e1 == null || e2 == null || e1.length != e2.length) {
        throw Exception("Invalid input lists for cosine distance calculation.");
      }

      List<double> e1AsDoubles = e1.cast<double>();
      List<double> e2AsDoubles = e2.cast<double>();

      double dotProduct = 0.0;
      double normE1 = 0.0;
      double normE2 = 0.0;

      // Calculate dot product and individual norms
      for (int i = 0; i < e1AsDoubles.length; i++) {
        dotProduct += e1AsDoubles[i] * e2AsDoubles[i];
        normE1 += e1AsDoubles[i] * e1AsDoubles[i];
        normE2 += e2AsDoubles[i] * e2AsDoubles[i];
      }

      normE1 = sqrt(normE1);
      normE2 = sqrt(normE2);

      // Check for potential division by zero
      if (normE1 * normE2 == 0) {
        return 0.0; // Or handle as appropriate
      }

      // Calculate cosine similarity and convert to distance
      return 1.0 - (dotProduct / (normE1 * normE2));
    } catch (error) {
      print("Error in cosine distance calculation: $error");
      // Handle the error appropriately, e.g., return a default value or throw a different exception
      return 0.0; // Example default value
    }
  }

  dispose() {}

  // ANTI SPOOFING

  Future initializeAntiSpoofingModel() async {
    // String modelPath = "assets/facebagnet.pth";

    try {
      // Load the anti-spoofing model

      // pytorch_mobile
      // this._antiSpoofingModel = await PyTorchMobile.loadModel(modelPath);

      // onnxruntime
      OrtEnv.instance.init();
      final sessionOptions = OrtSessionOptions();
      const assetFileName = 'assets/FaceBagNet_color_96.onnx';
      final rawAssetFile = await rootBundle.load(assetFileName);
      final bytes = rawAssetFile.buffer.asUint8List();
      final session = OrtSession.fromBuffer(bytes, sessionOptions);
    } catch (e) {
      print('Failed to load anti-spoofing model.');
      print(e);
    }
  }

  Future<List<double>?> isFaceSpoofedWithModel(
      CameraImage cameraImage, Face? face) async {
    print("===> isFaceSpoofedWithModel Starts");

    try {
      // Assuming you have a method to load the model from assets
      final session =
          await loadModelFromAssets('assets/FaceBagNet_color_96.onnx');

      // Check if face is null before passing it to _preProcess
      if (face == null) throw Exception('Face is null');

      List input = _preProcess(cameraImage, face);

      // Create OrtValue from input
      final shape = [1, 3, 96, 96];
      final inputOrt = OrtValueTensor.createTensorWithDataList(input, shape);

      // Create inputs map
      final inputs = {'onnx::Gather_0': inputOrt};

      // Create run options
      final runOptions = OrtRunOptions();

      // Run the model
      final outputs = session.run(runOptions, inputs);

      final FAStensor = outputs[0]?.value;

      print("===> outputs.length: " + outputs.length.toString());
      print("===> FAStensor: " + FAStensor.toString());

      List<double> FASTensorList = [];

      if (FAStensor != null &&
          FAStensor is List<List<double>> &&
          FAStensor.isNotEmpty) {
        FASTensorList = FAStensor[0];
      }

      // use softmax to get probabilities of the FASTensorList
      List<double> probabilities = softmax(FASTensorList);
      // probabilities = [0.6734, 0.3266];
      // probabilities = [0.3349, 0.6651];
      print("===> probabilities: " +
          probabilities.toString()); // prints the probabilities

      // release onnx components
      inputOrt.release();
      runOptions.release();
      session.release();
      print("===> isFaceSpoofedWithModel Ends");
      return probabilities;
    } catch (e) {
      print('An error occurred: $e');
      return null;
    }
  }

  List<List<List<int>>> convertImageToList(imglib.Image image) {
    int numChannels = 4; // Assuming the image is in RGBA format

    List<List<List<int>>> convertedImage = List.generate(
        image.height,
        (_) => List.generate(
            image.width, (_) => List<int>.filled(numChannels, 0)));

    for (int i = 0; i < image.height; i++) {
      for (int j = 0; j < image.width; j++) {
        int pixel = image.getPixelSafe(i, j);
        convertedImage[i][j][0] = imglib.getRed(pixel);
        convertedImage[i][j][1] = imglib.getGreen(pixel);
        convertedImage[i][j][2] = imglib.getBlue(pixel);
        convertedImage[i][j][3] = imglib.getAlpha(pixel);
      }
    }

    return convertedImage;
  }

  List<List<List<int>>> transposeImage(List<List<List<int>>> image) {
    int width = image.length;
    int height = image[0].length;
    int depth = image[0][0].length;

    List<List<List<int>>> transposedImage = List.generate(
      depth,
      (_) => List.generate(
        width,
        (_) => List<int>.filled(height, 0),
      ),
    );

    for (int i = 0; i < width; i++) {
      for (int j = 0; j < height; j++) {
        for (int k = 0; k < depth; k++) {
          transposedImage[k][i][j] = image[i][j][k];
        }
      }
    }

    return transposedImage;
  }

  List<double> softmax(List<double> scores) {
    double maxScore = scores.reduce(max);
    List<double> expScores =
        scores.map((score) => exp(score - maxScore)).toList();
    double sumExpScores = expScores.reduce((a, b) => a + b);
    return expScores.map((score) => score / sumExpScores).toList();
  }
}

extension Precision on double {
  double toFloat() {
    return double.parse(this.toStringAsFixed(2));
  }
}
