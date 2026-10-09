import 'package:flutter/material.dart';

import '../../api/api_client.dart';
import '../../state/app_state.dart';
import '../../widgets/common.dart';
import 'assistance_screen.dart';
import 'monitoring_screen.dart';

class DriverDashboard extends StatefulWidget {
  const DriverDashboard({super.key});

  @override
  State<DriverDashboard> createState() => _DriverDashboardState();
}

class _DriverDashboardState extends State<DriverDashboard> {
  Map<String, dynamic>? _request;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final state = AppScope.read(context);
    try {
      final body = await state.api.get('/assistance/requests/current');
      await state.refreshUser();
      if (mounted) {
        setState(() {
          _request = body['request'] as Map<String, dynamic>?;
          _error = null;
        });
      }
    } on ApiException catch (e) {
      if (mounted) setState(() => _error = e.message);
    }
  }

  Future<void> _open(Widget screen) async {
    await Navigator.push(context, MaterialPageRoute(builder: (_) => screen));
    if (mounted) _load();
  }

  @override
  Widget build(BuildContext context) {
    final user = AppScope.of(context).user!;
    final vehicle = user.driverProfile;
    final status = _request?['status'] as String?;
    final active = const ['REQUESTED', 'MATCHING', 'ACCEPTED', 'DRIVER_CONNECTED'].contains(status);
    return RefreshIndicator(
      onRefresh: _load,
      child: ListView(padding: const EdgeInsets.all(16), children: [
        Text('Hello, ${user.username}', style: Theme.of(context).textTheme.headlineSmall),
        Text('${user.cooperativeName} · ${vehicle?['vehicle_plate_number'] ?? 'no plate yet'}'),
        const SizedBox(height: 8),
        if (_error != null) Text(_error!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
        SectionCard(
          title: 'Account verification',
          icon: Icons.verified_user,
          trailing: StatusChip(user.verificationStatus),
          children: [
            Text(user.verificationStatus == 'VERIFIED'
                ? 'Your cooperative manager verified your account.'
                : 'Your cooperative manager has not verified your account yet. You can still monitor and '
                    'request assistance.'),
          ],
        ),
        SectionCard(title: 'Driver monitoring', icon: Icons.videocam, children: [
          const Text('Check for alcohol-related visual patterns with your phone\'s front camera, or control '
              'the in-vehicle camera connected to the SafeDrive server.'),
          const SizedBox(height: 8),
          FilledButton.icon(
            key: const Key('open-monitoring'),
            onPressed: () => _open(const MonitoringScreen()),
            icon: const Icon(Icons.play_arrow),
            label: const Text('Open monitoring'),
          ),
        ]),
        SectionCard(
          title: 'Umusare assistance',
          icon: Icons.support,
          trailing: status != null ? StatusChip(status) : null,
          children: [
            Text(active
                ? 'Assistance ${_request!['assistance_id']} is in progress.'
                : 'Need a safe driver? Request a verified Umusare near you. Your location is shared only '
                    'when you send the request.'),
            const SizedBox(height: 8),
            FilledButton.icon(
              key: const Key('open-assistance'),
              style: active ? null : FilledButton.styleFrom(backgroundColor: Colors.red[700]),
              onPressed: () => _open(const AssistanceScreen()),
              icon: Icon(active ? Icons.map : Icons.sos),
              label: Text(active ? 'Track assistance' : 'Request assistance'),
            ),
          ],
        ),
        const SizedBox(height: 8),
        const Disclaimer(),
      ]),
    );
  }
}
