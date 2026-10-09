import 'package:flutter/material.dart';

import '../../api/api_client.dart';
import '../../state/app_state.dart';
import '../../widgets/common.dart';

/// Cooperative manager: own cooperative only (scoped by the server), member verification decisions.
class ManagerDashboard extends StatefulWidget {
  const ManagerDashboard({super.key});

  @override
  State<ManagerDashboard> createState() => _ManagerDashboardState();
}

class _ManagerDashboardState extends State<ManagerDashboard> {
  Map<String, dynamic>? _body;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final body = await AppScope.read(context).api.get('/manager/console');
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

  Future<void> _review(Map<String, dynamic> member) async {
    final note = TextEditingController();
    final action = await showModalBottomSheet<String>(
      context: context,
      isScrollControlled: true,
      builder: (ctx) => Padding(
        padding: EdgeInsets.fromLTRB(16, 16, 16, MediaQuery.of(ctx).viewInsets.bottom + 16),
        child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Text('${member['username']} (${member['code']})', style: Theme.of(ctx).textTheme.titleMedium),
          Text('Status: ${member['verification_status']} · email ${member['email_verified'] == true ? 'verified' : 'not verified'}'
              '${member['vehicle_plate_number'] != null ? ' · plate ${member['vehicle_plate_number']}' : ''}'),
          const SizedBox(height: 8),
          TextField(controller: note, maxLength: 255,
              decoration: const InputDecoration(labelText: 'Note (required to reject or request information)')),
          FilledButton(onPressed: () => Navigator.pop(ctx, 'VERIFY'), child: const Text('Verify')),
          OutlinedButton(onPressed: () => Navigator.pop(ctx, 'REQUEST_INFO'), child: const Text('Request information')),
          OutlinedButton(onPressed: () => Navigator.pop(ctx, 'REJECT'), child: const Text('Reject')),
          TextButton(onPressed: () => Navigator.pop(ctx, 'SUSPEND'), child: const Text('Suspend')),
        ]),
      ),
    );
    if (action == null || !mounted) return;
    try {
      final result = await AppScope.read(context)
          .api
          .post('/manager/members/${member['id']}/review', {'action': action, 'note': note.text});
      if (mounted) showMessage(context, 'Member is now ${result['verification_status']}.');
      await _load();
    } on ApiException catch (e) {
      if (mounted) showMessage(context, e.message, error: true);
    }
  }

  Widget _memberTile(Map<String, dynamic> m) => ListTile(
        contentPadding: EdgeInsets.zero,
        title: Text('${m['username']}'),
        subtitle: Text('${m['code']} · ${m['operational']}${m['group_name'] != null ? ' · ${m['group_name']}' : ''}'),
        trailing: StatusChip(m['verification_status'] as String?),
        onTap: () => _review(m),
      );

  @override
  Widget build(BuildContext context) {
    if (_body == null) {
      return _error != null
          ? ErrorRetry(message: _error!, onRetry: _load)
          : const Center(child: CircularProgressIndicator());
    }
    final console = _body!['console'] as Map<String, dynamic>?;
    if (console == null) {
      return ErrorRetry(message: '${_body!['message']}', onRetry: _load);
    }
    final coop = console['cooperative'] as Map<String, dynamic>;
    final stats = console['stats'] as Map<String, dynamic>;
    final drivers = (console['drivers'] as List).cast<Map<String, dynamic>>();
    final umusare = (console['umusare'] as List).cast<Map<String, dynamic>>();
    final active = (console['active'] as List).cast<Map<String, dynamic>>();
    return RefreshIndicator(
      onRefresh: _load,
      child: ListView(padding: const EdgeInsets.all(16), children: [
        Text('${coop['name']}', style: Theme.of(context).textTheme.headlineSmall),
        Text('${coop['code']}${coop['district'] != null ? ' · ${coop['district']}' : ''}'),
        const SizedBox(height: 8),
        Wrap(spacing: 8, runSpacing: 8, children: [
          for (final e in stats.entries) Chip(label: Text('${e.key.replaceAll('_', ' ')}: ${e.value}')),
        ]),
        SectionCard(title: 'Active assistance (${active.length})', icon: Icons.support, children: [
          if (active.isEmpty) const Text('None right now.'),
          for (final a in active)
            ListTile(
              contentPadding: EdgeInsets.zero,
              title: Text('${a['code']} · ${a['driver']}'),
              subtitle: Text('${a['type_label']} · area ${a['approx_area']}${a['umusare'] != null ? ' · ${a['umusare']}' : ''}'),
              trailing: StatusChip(a['status'] as String?),
            ),
        ]),
        SectionCard(title: 'Drivers (${drivers.length})', icon: Icons.directions_car,
            children: [if (drivers.isEmpty) const Text('No drivers yet.'), ...drivers.map(_memberTile)]),
        SectionCard(title: 'Umusare (${umusare.length})', icon: Icons.support_agent,
            children: [if (umusare.isEmpty) const Text('No Umusare yet.'), ...umusare.map(_memberTile)]),
        Text('Tap a member to verify, reject, request information or suspend. Groups, chat and other '
            'management remain in the web console.', style: Theme.of(context).textTheme.bodySmall),
      ]),
    );
  }
}
