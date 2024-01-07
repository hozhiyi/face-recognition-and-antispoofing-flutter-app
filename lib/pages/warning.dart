import 'package:face_net_authentication/pages/issue_captured.dart';
import 'package:flutter/material.dart';

class WarningPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Warning"),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Icon(
              Icons.warning,
              size: 100,
              color: Colors.red,
            ),
            SizedBox(height: 20),
            Text(
              "Person Not Recognized",
              style: TextStyle(
                fontSize: 20,
                color: Colors.red,
              ),
            ),
            SizedBox(height: 10),
            Text(
              "Please try again or contact an administrator.",
              style: TextStyle(fontSize: 16),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                // You can add navigation logic here to go back or perform other actions.
                Navigator.pop(
                    context); // This will navigate back to the previous screen.
              },
              child: Text("OK"),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (BuildContext context) => IssueCapturedPage(),
                  ),
                );
              },
              child: Text("Report Issue"),
            ),
          ],
        ),
      ),
    );
  }
}
