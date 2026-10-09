import 'package:flutter/material.dart';

import '../../api/api_client.dart';
import '../../state/app_state.dart';
import '../../widgets/common.dart';
import '../driver/monitoring_screen.dart';

/// Administrator: read-only overview. User, cooperative and pricing management stay in the web console.
class AdminDashboard extends StatefulWidget {
  const AdminDashboard({super.key});

  @override
  State<AdminDashboard> createState() => _AdminDashboardState();
}

class _AdminDashboardState extends State<AdminDashboard> {
  Map<String, dynamic>? _body;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final body = await AppScope.read(context).api.get('/admin/overview');
      if (mounted) {
        setState(() {
          _body = body;
          _error = null;
        });
      }
    } on ApiException catch (e) {
      if (mounted) setState(() => _error = e.message);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_body == null) {
      return _error != null
          ? ErrorRetry(message: _error!, onRetry: _load)
          : const Center(child: CircularProgressIndicator());
    }
    final users = _body!['users'] as Map<String, dynamic>;
    final coops = _body!['cooperatives'] as Map<String, dynamic>;
    final assistance = (_body!['assistance'] as List).cast<Map<String, dynamic>>();
    return RefreshIndicator(
      onRefresh: _load,
      child: ListView(padding: const EdgeInsets.all(16), children: [
        SectionCard(title: 'Users', icon: Icons.people, children: [
          Wrap(spacing: 8, children: [for (final e in users.entries) Chip(label: Text('${e.key}: ${e.value}'))]),
        ]),
        SectionCard(title: 'Cooperatives', icon: Icons.apartment, children: [
          Wrap(spacing: 8, children: [for (final e in coops.entries) Chip(label: Text('${e.key}: ${e.value}'))]),
        ]),
        SectionCard(title: 'Assistance (latest ${assistance.length})', icon: Icons.support, children: [
          if (assistance.isEmpty) const Text('No assistance requests yet.'),
          for (final a in assistance)
            ListTile(
              contentPadding: EdgeInsets.zero,
              title: Text('AS-${'${a['id']}'.padLeft(6, '0')} · ${a['driver']}'),
              subtitle: Text('${a['cooperative'] ?? '—'} · area ${a['approx_area']}'
                  '${a['umusare'] != null ? ' · ${a['umusare']}' : ''}'),
              trailing: StatusChip(a['status'] as String?),
            ),
        ]),
        OutlinedButton.icon(
          onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const MonitoringScreen())),
          icon: const Icon(Icons.videocam),
          label: const Text('Monitoring (support / research)'),
        ),
        const SizedBox(height: 8),
        Text('Use the web console to manage users, cooperatives, managers and pricing.',
            style: Theme.of(context).textTheme.bodySmall),
      ]),
    );
  }
}
