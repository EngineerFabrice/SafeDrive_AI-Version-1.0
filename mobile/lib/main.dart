import 'package:flutter/material.dart';

import 'config.dart';
import 'screens/auth/login_screen.dart';
import 'screens/home_screen.dart';
import 'state/app_state.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  final state = AppState()..init();
  runApp(SafeDriveApp(state: state));
}

class SafeDriveApp extends StatelessWidget {
  const SafeDriveApp({super.key, required this.state});

  final AppState state;

  @override
  Widget build(BuildContext context) {
    return AppScope(
      state: state,
      child: MaterialApp(
        title: kAppName,
        debugShowCheckedModeBanner: false,
        theme: ThemeData(colorSchemeSeed: const Color(0xFF0D47A1), useMaterial3: true),
        darkTheme: ThemeData(colorSchemeSeed: const Color(0xFF0D47A1), brightness: Brightness.dark, useMaterial3: true),
        home: const _Root(),
      ),
    );
  }
}

class _Root extends StatelessWidget {
  const _Root();

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    if (state.initializing) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    // A new key per user forces a fresh dashboard (and its timers) after every sign-in / sign-out.
    return state.user == null
        ? const LoginScreen(key: ValueKey('login'))
        : HomeScreen(key: ValueKey('home-${state.user!.id}'));
  }
}
