import 'dart:ui';

import 'package:camera/camera.dart';
import 'package:google_ml_kit/google_ml_kit.dart';

class CameraService {
  CameraController? _cameraController;
  CameraController? get cameraController => this._cameraController;

  InputImageRotation? _cameraRotation;
  InputImageRotation? get cameraRotation => this._cameraRotation;

  String? _imagePath;
  String? get imagePath => this._imagePath;

  Future<void> initialize() async { 
    if (_cameraController != null) return;
    CameraDescription description = await _getCameraDescription();
    await _setupCameraController(description: description);
    this._cameraRotation = rotationIntToImageRotation(
      description.sensorOrientation,
    );
  }

  Future<CameraDescription> _getCameraDescription() async {
    // fetches available cameras and selects the front-facing camera.
    List<CameraDescription> cameras = await availableCameras();
    return cameras.firstWhere((CameraDescription camera) =>
        camera.lensDirection == CameraLensDirection.front);
  }

  Future _setupCameraController({
    // sets up the camera controller with the provided camera description and initializes it.
    required CameraDescription description,
  }) async {
    this._cameraController = CameraController(
      description,
      ResolutionPreset.high,
      enableAudio: false,
    );
    await _cameraController?.initialize();
  }

  InputImageRotation rotationIntToImageRotation(int rotation) {
    // converts integer rotation values to InputImageRotation enum values.
    switch (rotation) {
      case 90:
        return InputImageRotation.rotation90deg;
      case 180:
        return InputImageRotation.rotation180deg;
      case 270:
        return InputImageRotation.rotation270deg;
      default:
        return InputImageRotation.rotation0deg;
    }
  }

  Future<XFile?> takePicture() async {
    // captures a picture using the camera controller, stores the image path,
    // returns an XFile representing the captured image file.
    assert(_cameraController != null, 'Camera controller not initialized');
    await _cameraController?.stopImageStream();
    XFile? file = await _cameraController?.takePicture();
    _imagePath = file?.path;
    return file;
  }

  Size getImageSize() {
    // retrieves the size of the image preview.
    assert(_cameraController != null, 'Camera controller not initialized');
    assert(
        _cameraController!.value.previewSize != null, 'Preview size is null');
    return Size(
      _cameraController!.value.previewSize!.height,
      _cameraController!.value.previewSize!.width,
    );
  }

  dispose() async {
    // disposes of the camera controller when it's no longer needed.
    await this._cameraController?.dispose();
    this._cameraController = null;
  }
}
