import 'dart:async';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:face_net_authentication/locator.dart';
import 'package:face_net_authentication/pages/models/user.model.dart';
import 'package:face_net_authentication/pages/warning.dart';
import 'package:face_net_authentication/pages/widgets/auth_button.dart';
import 'package:face_net_authentication/pages/widgets/camera_detection_preview.dart';
import 'package:face_net_authentication/pages/widgets/camera_header.dart';
import 'package:face_net_authentication/pages/widgets/signin_form.dart';
import 'package:face_net_authentication/pages/widgets/single_picture.dart';
import 'package:face_net_authentication/services/camera.service.dart';
import 'package:face_net_authentication/services/face_detector_service.dart';
import 'package:face_net_authentication/services/ml_service.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as imglib;
// import 'package:image_picker/image_picker.dart';

class SignIn extends StatefulWidget {
  const SignIn({Key? key}) : super(key: key);

  @override
  SignInState createState() => SignInState();
}

class SignInState extends State<SignIn> {
  CameraService _cameraService = locator<CameraService>();
  FaceDetectorService _faceDetectorService = locator<FaceDetectorService>();
  MLService _mlService = locator<MLService>();
  // late imglib.Image processedImage;
  // initialize processedImage
  imglib.Image processedImage = imglib.Image(0, 0);

  GlobalKey<ScaffoldState> scaffoldKey = GlobalKey<ScaffoldState>();

  bool _isPictureTaken = false;
  bool _isInitializing = false;
  Future<List<double>?> FASoutputs = Future.value([]);

  @override
  void initState() {
    super.initState();
    _start();
  }

  @override
  void dispose() {
    _cameraService.dispose();
    _mlService.dispose();
    _faceDetectorService.dispose();
    super.dispose();
  }

  Future _start() async {
    setState(() => _isInitializing = true);
    await _cameraService.initialize();
    setState(() => _isInitializing = false);
    _frameFaces();
  }

  _frameFaces() async {
    bool processing = false;
    _cameraService.cameraController!
        .startImageStream((CameraImage image) async {
      if (processing) return; // prevents unnecessary overprocessing.
      processing = true;
      await _predictFacesFromImage(image: image);
      processing = false;
    });
  }

  Future<void> _predictFacesFromImage({@required CameraImage? image}) async {
    assert(image != null, 'Image is null');

    await _faceDetectorService.detectFacesFromImage(image!);

    if (_faceDetectorService.faceDetected) {
      // FASoutputs = _mlService.isFaceSpoofedWithModel(
      //     image, _faceDetectorService.faces[0]);

      final results = await _mlService.MODIFIEDisFaceSpoofedWithModel(
          image, _faceDetectorService.faces[0]);

      if (results != null) {
        // Access processedImage and probabilities from the list
        processedImage = results[0] as imglib.Image;
        final FASoutputs = results[1] as List<double>;

        _mlService.setCurrentPrediction(image, _faceDetectorService.faces[0]);
      } else {
        // Handle the case where results are null (e.g., an error occurred)
        print('Error: No results returned from MODIFIEDisFaceSpoofedWithModel');
      }
    }

    if (mounted) setState(() {});
  }

  Future<void> takePicture() async {
    if (_faceDetectorService.faceDetected) {
      await _cameraService.takePicture();
      setState(() => _isPictureTaken = true);
    } else {
      showDialog(
          context: context,
          builder: (context) =>
              AlertDialog(content: Text('No face detected!')));
    }
  }

  _onBackPressed() {
    Navigator.of(context).pop();
  }

  _reload() {
    if (mounted) setState(() => _isPictureTaken = false);
    _start();
  }

  // // ORIGINAL
  // Future<void> onTap() async {
  //   await takePicture();

  //   // if the probability of FASoutputs then return warning page
  //   // else continue to sign in page
  //   final outputs = await FASoutputs;
  //   if (outputs != null && outputs.isNotEmpty && outputs[0] > 0.5) {
  //     // display a text box saying that the user is a spoof
  //     showDialog(
  //       context: context,
  //       builder: (context) => AlertDialog(
  //         content: Text('Spoof detected!'),
  //       ),
  //     );
  //   } else {
  //     if (_faceDetectorService.faceDetected) {
  //       User? user = await _mlService.predict();
  //       if (user != null) {
  //         var bottomSheetController = scaffoldKey.currentState!
  //             .showBottomSheet((context) => signInSheet(user: user));
  //         bottomSheetController.closed.whenComplete(_reload);
  //       } else {
  //         // If user is null, navigate to another page
  //         Navigator.of(context).push(
  //           MaterialPageRoute(
  //             builder: (context) => WarningPage(),
  //           ),
  //         );
  //       }
  //     }
  //   }
  // }

  // MODIFIED
  Future<void> onTap() async {
    await takePicture();

    // if the probability of FASoutputs then return warning page
    // else continue to sign in page
    final outputs = await FASoutputs;
    if (outputs != null && outputs.isNotEmpty && outputs[0] > 0.5) {
      // display a text box saying that the user is a spoof
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          content: Text('Spoof detected!'),
        ),
      );
    } else {
      if (_faceDetectorService.faceDetected) {
        User? user = await _mlService.predict();
        if (user != null) {
          // Display processedImage in a popup dialog
          showDialog(
            context: context,
            builder: (context) => Dialog(
              child: Container(
                padding: EdgeInsets.all(20),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    if (processedImage != null)
                      Image.memory(
                          Uint8List.fromList(imglib.encodePng(processedImage))),
                    Text('Processed Image'),
                    ElevatedButton(
                      onPressed: () => Navigator.of(context).pop(),
                      child: Text('Close'),
                    ),
                  ],
                ),
              ),
            ),
          );
          var bottomSheetController = scaffoldKey.currentState!
              .showBottomSheet((context) => signInSheet(user: user));
          bottomSheetController.closed.whenComplete(_reload);
        } else {
          // If user is null, navigate to another page
          Navigator.of(context).push(
            MaterialPageRoute(
              builder: (context) => WarningPage(),
            ),
          );
        }
      }
    }
  }

  Widget getBodyWidget() {
    if (_isInitializing) return Center(child: CircularProgressIndicator());
    if (_isPictureTaken)
      return SinglePicture(imagePath: _cameraService.imagePath!);
    return CameraDetectionPreview();
  }

  @override
  Widget build(BuildContext context) {
    Widget header = CameraHeader("Clock In", onBackPressed: _onBackPressed);
    Widget body = getBodyWidget();
    Widget? fab;
    if (!_isPictureTaken) fab = AuthButton(onTap: onTap);

    return Scaffold(
      key: scaffoldKey,
      body: Stack(
        children: [body, header],
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
      floatingActionButton: fab,
    );
  }

  signInSheet({@required User? user}) => user == null
      ? Container(
          width: MediaQuery.of(context).size.width,
          padding: EdgeInsets.all(20),
          child: Text(
            'User not found 😞',
            style: TextStyle(fontSize: 20),
          ),
        )
      : SignInSheet(user: user);
}
