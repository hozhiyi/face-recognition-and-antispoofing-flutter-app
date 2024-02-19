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
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart' as path;
import 'package:tflite_flutter/tflite_flutter.dart';

class MLService {
  Interpreter? _interpreter;
  double threshold = 0.5;
  List _predictedData = [];
  List get predictedData => _predictedData;
  double dist = 0;

  // = = = = = = = = = = = //
  //  ANTI SPOOFING (onnx) //
  // = = = = = = = = = = = //
  Future<OrtSession> loadModelFromAssets() async {
    OrtEnv.instance.init();
    final sessionOptions = OrtSessionOptions();
    // const assetFileName = 'assets/FaceBagNet_color_96.onnx';
    // const assetFileName = 'assets/AntiSpoofing_print-replay_128.onnx';
    const assetFileName = 'assets/2.7_80x80_MiniFASNetV2.onnx';

    final rawAssetFile = await rootBundle.load(assetFileName);
    final bytes = rawAssetFile.buffer.asUint8List();
    final session = OrtSession.fromBuffer(bytes, sessionOptions);

    return session;
  }

  // // ORIGINAL
  // Future<List<double>?> isFaceSpoofedWithModel(
  //     CameraImage cameraImage, Face? face) async {
  //   print("===> isFaceSpoofedWithModel Starts");

  //   try {
  //     // Assuming you have a method to load the model from assets
  //     final session = await loadModelFromAssets();

  //     if (face == null) throw Exception('Face is null');

  //     Future<List> input = _preProcess(cameraImage, face);

  //     // Create OrtValue from input
  //     // final shape = [1, 3, 128, 128];
  //     final shape = [1, 3, 80, 80];
  //     // final shape = [1, 80, 80, 3];
  //     final inputOrt =
  //         OrtValueTensor.createTensorWithDataList(await input, shape);
  //     final inputName = session.inputNames[0];

  //     // Create inputs map
  //     final inputs = {inputName: inputOrt};

  //     // Create run options
  //     final runOptions = OrtRunOptions();

  //     // Run the model
  //     final outputs = session.run(runOptions, inputs);

  //     final FAStensor = outputs[0]?.value;

  //     print("===> outputs.length: " + outputs.length.toString());
  //     print("===> FAStensor: " + FAStensor.toString());

  //     List<double> FASTensorList = [];

  //     if (FAStensor != null &&
  //         FAStensor is List<List<double>> &&
  //         FAStensor.isNotEmpty) {
  //       FASTensorList = FAStensor[0];
  //     }

  //     // use softmax to get probabilities of the FASTensorList
  //     List<double> probabilities = softmax(FASTensorList);
  //     // probabilities = [0.6734, 0.3266];
  //     // probabilities = [0.3349, 0.6651];
  //     print("===> probabilities: " +
  //         probabilities.toString()); // prints the probabilities

  //     // release onnx components
  //     inputOrt.release();
  //     runOptions.release();
  //     session.release();
  //     print("===> isFaceSpoofedWithModel Ends");
  //     return probabilities;
  //   } catch (e) {
  //     print('An error occurred: $e');
  //     return null;
  //   }
  // }

  // MODIFIED
  Future<List?> MODIFIEDisFaceSpoofedWithModel(
      CameraImage cameraImage, Face? face) async {
    print("===> isFaceSpoofedWithModel Starts");

    try {
      // Assuming you have a method to load the model from assets
      final session = await loadModelFromAssets();

      if (face == null) throw Exception('Face is null');

      final processedData = await _MODIFIEDpreProcess(cameraImage, face);
      // Access img and imageAsList from the returned list
      final processedImg = processedData[0] as imglib.Image;
      final input = processedData[1] as Float32List;

      // Create OrtValue from input
      // final shape = [1, 3, 128, 128];
      final shape = [1, 3, 80, 80];
      // final shape = [1, 80, 80, 3];
      final inputOrt =
          OrtValueTensor.createTensorWithDataList(await input, shape);
      final inputName = session.inputNames[0];

      // Create inputs map
      final inputs = {inputName: inputOrt};

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
      return [processedImg, probabilities];
    } catch (e) {
      print('An error occurred: $e');
      return null;
    }
  }

  // // ORIGINAL
  // Future<List> _preProcess(CameraImage image, Face faceDetected) async {
  //   imglib.Image croppedImage = _cropFace(image, faceDetected);

  //   imglib.Image img;
  //   // img = imglib.copyResizeCropSquare(croppedImage, 128);
  //   img = imglib.copyResizeCropSquare(croppedImage, 80);
  //   final directory = await path.getApplicationDocumentsDirectory();
  //   final file = File(join(directory.path, 'resized.png'));
  //   // print the image path
  //   print(file.path);
  //   await file.writeAsBytes(imglib.encodePng(img));
  //   // new File("resized.png").writeAsBytesSync(imglib.encodePng(img));

  //   // Float32List imageAsList = imageToByteListFloat32(croppedImage, 128);
  //   Float32List imageAsList = imageToByteListFloat32(croppedImage, 80);

  //   // normalization
  //   for (int i = 0; i < imageAsList.length; i++) {
  //     imageAsList[i] = imageAsList[i];
  //   }
  //   return imageAsList;
  // }

  // MODIFIED
  Future<List> _MODIFIEDpreProcess(CameraImage image, Face faceDetected) async {
    imglib.Image croppedImage = _cropFace(image, faceDetected);

    imglib.Image img;
    // img = imglib.copyResizeCropSquare(croppedImage, 128);
    img = imglib.copyResizeCropSquare(croppedImage, 80);
    // final directory = await path.getApplicationDocumentsDirectory();
    // final file = File(join(directory.path, 'resized.png'));
    // await file.writeAsBytes(imglib.encodePng(img));
    final directory = await path.getExternalStorageDirectory();
    final file = File(join(directory!.path, 'resized.png'));
    try {
      await file.writeAsBytes(imglib.encodePng(img));
      print("===> file.path: " + file.path);
    } catch (e) {
      print('Error: $e');
    }

    // Float32List imageAsList = imageToByteListFloat32(croppedImage, 128);
    Float32List imageAsList = imageToByteListFloat32(croppedImage, 80);

    // normalization
    for (int i = 0; i < imageAsList.length; i++) {
      imageAsList[i] = imageAsList[i];
    }
    return [img, imageAsList];
  }

  // = = = = = = = = = = = = = = = //
  //   FACE RECOGNITION  (TFLITE)  //
  // = = = = = = = = = = = = = = = //
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

  Future<void> setCurrentPrediction(CameraImage cameraImage, Face? face) async {
    if (_interpreter == null) throw Exception('Interpreter is null');
    if (face == null) throw Exception('Face is null');

    // List input = _preProcess(cameraImage, face);
    // _preProcess starts
    imglib.Image croppedImage = _cropFace(cameraImage, face);

    // save the cropped image
    // await saveImage('assets/croppedImage-fr.jpg', croppedImage);

    imglib.Image img;
    img = imglib.copyResizeCropSquare(croppedImage, 112);
    List input = imageToByteListFloat32(img, 112);
    // _preProcess ends

    input = input.reshape([1, 112, 112, 3]);
    print("==> input : " + input.toString());
    List output = List.generate(1, (index) => List.filled(256, 0));

    this._interpreter?.run(input, output);
    // print("==> ori output : " + output.toString());

    output = output.reshape([256]);
    // print("==> mod output : " + output.toString());

    this._predictedData = List.from(output);
  }

  Future<User?> predict() async {
    // print("===> this._predictedData" + this._predictedData.toString());
    return _searchResult(this._predictedData);
  }

  List<double> softmax(List<double> scores) {
    double maxScore = scores.reduce(max);
    List<double> expScores =
        scores.map((score) => exp(score - maxScore)).toList();
    double sumExpScores = expScores.reduce((a, b) => a + b);
    return expScores.map((score) => score / sumExpScores).toList();
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

  Float32List imageToByteListFloat32(imglib.Image image, int imageSize) {
    var convertedBytes = Float32List(1 * imageSize * imageSize * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;

    for (var i = 0; i < imageSize; i++) {
      for (var j = 0; j < imageSize; j++) {
        var pixel = image.getPixel(j, i);
        // buffer[pixelIndex++] = (imglib.getRed(pixel) - 128) / 128;
        // buffer[pixelIndex++] = (imglib.getGreen(pixel) - 128) / 128;
        // buffer[pixelIndex++] = (imglib.getBlue(pixel) - 128) / 128;
        // buffer[pixelIndex++] = imglib.getRed(pixel) / 255;
        // buffer[pixelIndex++] = imglib.getGreen(pixel) / 255;
        // buffer[pixelIndex++] = imglib.getBlue(pixel) / 255;
        buffer[pixelIndex++] = imglib.getRed(pixel) / 1;
        buffer[pixelIndex++] = imglib.getGreen(pixel) / 1;
        buffer[pixelIndex++] = imglib.getBlue(pixel) / 1;
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

  dispose() {
    _interpreter?.close();
  }

  // Future<void> saveImage(filePath, imglib.Image image) async {
  //   // Encode the image as JPG
  //   List<int> jpgBytes = imglib.encodeJpg(image);

  //   // Write the bytes to a file
  //   await File(filePath).writeAsBytes(jpgBytes);
  // }
}

extension Precision on double {
  double toFloat() {
    return double.parse(this.toStringAsFixed(2));
  }
}
