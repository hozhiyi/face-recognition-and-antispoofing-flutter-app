import 'package:flutter/material.dart';
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';

class IssueCapturedPage extends StatefulWidget {
  @override
  _IssueCapturedPageState createState() => _IssueCapturedPageState();
}

class _IssueCapturedPageState extends State<IssueCapturedPage> {
  final dbHelper = DatabaseHelper.instance;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Issue Captured"),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Icon(
              Icons.check_circle,
              size: 100,
              color: Colors.green,
            ),
            SizedBox(height: 20),
            Text(
              "Issue Captured and Noted",
              style: TextStyle(
                fontSize: 20,
                color: Colors.green,
              ),
            ),
            SizedBox(height: 10),
            Text(
              "The issue has been recorded successfully.",
              style: TextStyle(fontSize: 16),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: () async {
                // Insert the issue into the SQLite database.
                await dbHelper.insertIssue('Your issue description');
                Navigator.pop(context); // This will navigate back to the previous screen.
              },
              child: Text("OK"),
            ),
          ],
        ),
      ),
    );
  }
}

class DatabaseHelper {
  DatabaseHelper._privateConstructor();
  static final DatabaseHelper instance = DatabaseHelper._privateConstructor();

  static Database? _database;

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDatabase();
    return _database!;
  }

  Future<Database> _initDatabase() async {
    final path = join(await getDatabasesPath(), 'issue_database.db');
    return await openDatabase(path, version: 1, onCreate: _createDatabase);
  }

  Future<void> _createDatabase(Database db, int version) async {
    await db.execute('''
      CREATE TABLE issues(
        id INTEGER PRIMARY KEY,
        description TEXT
      )
    ''');
  }

  Future<int> insertIssue(String description) async {
    final db = await database;
    return await db.insert('issues', {'description': description});
  }
}
