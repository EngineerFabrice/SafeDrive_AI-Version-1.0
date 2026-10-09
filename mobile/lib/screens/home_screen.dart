import 'package:flutter/material.dart';

import '../api/api_client.dart';
import '../state/app_state.dart';
import '../widgets/common.dart';
import 'about_screen.dart';
import 'admin/admin_dashboard.dart';
import 'auth/verify_email_screen.dart';
import 'driver/driver_dashboard.dart';
import 'manager/manager_dashboard.dart';
import 'notifications_screen.dart';
import 'profile_screen.dart';
import 'umusare/umusare_dashboard.dart';

/// Picks the dashboard for the signed-in user's role (the role always comes from the server).
class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    final user = state.user!;
    final Widget body = switch (user.role) {
      'driver' => const DriverDashboard(),
      'umusare' => const UmusareDashboard(),
      'manager' => const ManagerDashboard(),
      'admin' => const AdminDashboard(),
      _ => const Center(child: Text('Unknown role')),
    };
    return Scaffold(
      appBar: AppBar(
        title: Text(user.roleLabel),
        actions: [
          IconButton(
            tooltip: 'Notifications',
            icon: const Icon(Icons.notifications),
            onPressed: () =>
                Navigator.push(context, MaterialPageRoute(builder: (_) => const NotificationsScreen())),
          ),
          PopupMenuButton<String>(
            onSelected: (v) async {
              switch (v) {
                case 'profile':
                  Navigator.push(context, MaterialPageRoute(builder: (_) => const ProfileScreen()));
                case 'about':
                  Navigator.push(context, MaterialPageRoute(builder: (_) => const AboutScreen()));
                case 'logout':
                  await AppScope.read(context).logout();
              }
            },
            itemBuilder: (_) => const [
              PopupMenuItem(value: 'profile', child: Text('My profile')),
              PopupMenuItem(value: 'about', child: Text('About')),
              PopupMenuItem(value: 'logout', child: Text('Sign out')),
            ],
          ),
        ],
      ),
      body: Column(children: [
        if (!user.emailVerified)
          MaterialBanner(
            content: const Text('Verify your email address to continue the cooperative verification.'),
            actions: [
              TextButton(
                onPressed: () =>
                    Navigator.push(context, MaterialPageRoute(builder: (_) => const VerifyEmailScreen())),
                child: const Text('VERIFY'),
              ),
            ],
          ),
        if (user.needsTerms)
          MaterialBanner(
            content: const Text('The Terms & Conditions or Privacy Policy changed. Please review and accept them.'),
            actions: [
              TextButton(
                onPressed: () async {
                  try {
                    final body = await state.api.post('/me/accept-terms', {'accept_terms': true});
                    state.updateUser(body['user'] as Map<String, dynamic>);
                  } on ApiException catch (e) {
                    if (context.mounted) showMessage(context, e.message, error: true);
                  }
                },
                child: const Text('ACCEPT'),
              ),
            ],
          ),
        Expanded(child: body),
      ]),
    );
  }
}
