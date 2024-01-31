import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

// a function to load onnx model that returns the ortsession and input_name
Future<OrtSession> loadFASModel(String model_path) async {
  // initialize the ort environment
  OrtEnv.instance.init();

  //  create the session
  final sessionOptions = OrtSessionOptions();

  // const assetFileName = model_path;
  final rawAssetFile = await rootBundle.load(model_path);
  final bytes = rawAssetFile.buffer.asUint8List();
  final session = OrtSession.fromBuffer(bytes, sessionOptions);
  return session;
}


// // a function to run the ortsession
// Future make_prediction(CameraImage cameraImage, Face? face) async {
//   try {
//     final session = await loadFASModel('assets/models/FAS.onnx');
//     if (face == null) throw Exception('No face detected!');
//     List input 

//   }
// }