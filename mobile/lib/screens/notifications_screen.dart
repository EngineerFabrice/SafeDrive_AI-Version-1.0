import 'package:flutter/material.dart';

import '../api/api_client.dart';
import '../state/app_state.dart';
import '../widgets/common.dart';

class NotificationsScreen extends StatefulWidget {
  const NotificationsScreen({super.key});

  @override
  State<NotificationsScreen> createState() => _NotificationsScreenState();
}

class _NotificationsScreenState extends State<NotificationsScreen> {
  List<Map<String, dynamic>>? _items;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final body = await AppScope.read(context).api.get('/notifications');
      if (mounted) {
        setState(() {
          _items = (body['notifications'] as List).cast<Map<String, dynamic>>();
          _error = null;
        });
      }
    } on ApiException catch (e) {
      if (mounted) setState(() => _error = e.message);
    }
  }

  Future<void> _markRead() async {
    try {
      await AppScope.read(context).api.post('/notifications/read');
      await _load();
    } on ApiException catch (e) {
      if (mounted) showMessage(context, e.message, error: true);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Notifications'), actions: [
        IconButton(tooltip: 'Mark all read', onPressed: _markRead, icon: const Icon(Icons.done_all)),
      ]),
      body: _error != null
          ? ErrorRetry(message: _error!, onRetry: _load)
          : _items == null
              ? const Center(child: CircularProgressIndicator())
              : RefreshIndicator(
                  onRefresh: _load,
                  child: _items!.isEmpty
                      ? ListView(children: const [
                          Padding(padding: EdgeInsets.all(32), child: Center(child: Text('No notifications.'))),
                        ])
                      : ListView(children: [
                          for (final n in _items!)
                            ListTile(
                              leading: Icon(n['unread'] == true ? Icons.circle : Icons.circle_outlined, size: 12),
                              title: Text('${n['title']}'),
                              subtitle: Text('${n['created_at']}'.replaceFirst('T', ' ').replaceFirst('Z', ' UTC')),
                            ),
                        ]),
                ),
    );
  }
}
