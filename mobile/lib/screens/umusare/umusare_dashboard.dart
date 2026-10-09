import 'dart:async';

import 'package:flutter/material.dart';

import '../../api/api_client.dart';
import '../../services/location_service.dart';
import '../../state/app_state.dart';
import '../../widgets/common.dart';

/// Umusare: availability (shares position for matching), incoming offers, active assistance with live
/// location, and payment confirmation. Polls the server every 5 seconds while open.
class UmusareDashboard extends StatefulWidget {
  const UmusareDashboard({super.key});

  @override
  State<UmusareDashboard> createState() => _UmusareDashboardState();
}

class _UmusareDashboardState extends State<UmusareDashboard> {
  Map<String, dynamic>? _view;
  String? _error;
  String? _locationError;
  bool _busy = false;
  Timer? _poll;
  Timer? _share;
  DateTime? _lastAvailabilityRefresh;
  late final ApiClient _api;

  @override
  void initState() {
    super.initState();
    _api = AppScope.read(context).api;
    _refresh();
    _poll = Timer.periodic(const Duration(seconds: 5), (_) => _refresh());
  }

  @override
  void dispose() {
    _poll?.cancel();
    _share?.cancel();
    super.dispose();
  }

  String? get _availability => (_view?['profile'] as Map?)?['availability'] as String?;
  Map<String, dynamic>? get _active => _view?['active'] as Map<String, dynamic>?;

  Future<void> _refresh() async {
    try {
      final body = await _api.get('/assistance/umusare/status');
      if (!mounted) return;
      setState(() {
        _view = body;
        _error = null;
      });
      _updateSharing();
    } on ApiException catch (e) {
      if (mounted) setState(() => _error = e.message);
    }
  }

  /// While AVAILABLE, the position used for matching is refreshed every few minutes (it must stay recent);
  /// during an accepted assistance, the live position is sent every 15 seconds to the matched driver only.
  void _updateSharing() {
    final active = _active;
    if (active != null && _share == null) {
      _shareLive();
      _share = Timer.periodic(const Duration(seconds: 15), (_) => _shareLive());
    } else if (active == null && _share != null) {
      _share!.cancel();
      _share = null;
    }
    if (_availability == 'AVAILABLE' &&
        (_lastAvailabilityRefresh == null ||
            DateTime.now().difference(_lastAvailabilityRefresh!) > const Duration(minutes: 5))) {
      _lastAvailabilityRefresh = DateTime.now();
      _setAvailable(true, quiet: true);
    }
  }

  Future<void> _shareLive() async {
    final id = _active?['id'];
    if (id == null) return;
    try {
      final fix = await LocationService.current();
      await _api.post('/assistance/requests/$id/location', fix.toJson());
      if (mounted && _locationError != null) setState(() => _locationError = null);
    } on LocationException catch (e) {
      if (mounted) setState(() => _locationError = e.message);
    } on ApiException catch (e) {
      if (e.code != 'SHARING_STOPPED' && mounted) setState(() => _locationError = e.message);
    }
  }

  Future<void> _setAvailable(bool available, {bool quiet = false}) async {
    if (!quiet) setState(() => _busy = true);
    try {
      final body = available
          ? await _api.post('/assistance/umusare/availability', {'available': true, ...(await LocationService.current()).toJson()})
          : await _api.post('/assistance/umusare/availability', {'available': false});
      if (available) _lastAvailabilityRefresh = DateTime.now();
      if (mounted) setState(() => _view = body['status'] as Map<String, dynamic>);
    } on LocationException catch (e) {
      if (mounted && !quiet) showMessage(context, e.message, error: true);
      if (mounted) setState(() => _locationError = e.message);
    } on ApiException catch (e) {
      if (mounted && !quiet) showMessage(context, e.message, error: true);
    } finally {
      if (mounted && !quiet) setState(() => _busy = false);
    }
  }

  Future<void> _act(String path, {Map<String, dynamic>? body, String? done}) async {
    setState(() => _busy = true);
    try {
      final result = await _api.post(path, body);
      if (!mounted) return;
      setState(() => _view = result.containsKey('profile') ? result : _view);
      if (done != null) showMessage(context, done);
      await _refresh();
    } on ApiException catch (e) {
      if (mounted) showMessage(context, e.message, error: true);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final user = AppScope.of(context).user!;
    final v = _view;
    if (v == null) {
      return _error != null
          ? ErrorRetry(message: _error!, onRetry: _refresh)
          : const Center(child: CircularProgressIndicator());
    }
    final profile = v['profile'] as Map<String, dynamic>;
    final verified = profile['verification'] == 'VERIFIED';
    final incoming = (v['incoming'] as List).cast<Map<String, dynamic>>();
    final payments = (v['payments'] as List).cast<Map<String, dynamic>>();
    final active = _active;
    return RefreshIndicator(
      onRefresh: _refresh,
      child: ListView(padding: const EdgeInsets.all(16), children: [
        Text('Hello, ${user.username}', style: Theme.of(context).textTheme.headlineSmall),
        Text('${profile['umusare_id']} · ${profile['cooperative'] ?? '—'}'),
        if (_error != null) Text(_error!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
        SectionCard(
          title: 'Availability',
          icon: Icons.toggle_on,
          trailing: StatusChip(_availability),
          children: [
            if (!verified)
              Text('Your cooperative manager must verify you (status: ${profile['verification']}) before you can '
                  'receive requests.'),
            if (user.phone == null || user.phone!.isEmpty)
              const Text('Add your phone number in My profile: drivers pay to this number.'),
            SwitchListTile(
              key: const Key('availability-switch'),
              contentPadding: EdgeInsets.zero,
              title: const Text('Available for assistance'),
              subtitle: const Text('Shares your current position with SafeDrive matching while available. '
                  'Going offline deletes it.'),
              value: _availability == 'AVAILABLE' || _availability == 'BUSY',
              onChanged: _busy || _availability == 'BUSY' || !verified ? null : (on) => _setAvailable(on),
            ),
            if (_locationError != null)
              Text(_locationError!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
          ],
        ),
        if (active != null) _activeCard(active),
        if (active == null)
          SectionCard(title: 'Incoming requests (${incoming.length})', icon: Icons.notifications_active, children: [
            if (incoming.isEmpty)
              Text(_availability == 'AVAILABLE' ? 'Waiting for requests near you…' : 'Go available to receive requests.'),
            for (final r in incoming)
              Card(
                color: Theme.of(context).colorScheme.surfaceContainerHighest,
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Text('${r['assistance_id']} · ${r['type_label']}', style: const TextStyle(fontWeight: FontWeight.bold)),
                    Text('${r['reason']} · ${r['approx_distance']}'),
                    Text('${(r['fare'] as Map?)?['rate_label'] ?? ''} — ${(r['fare'] as Map?)?['note'] ?? ''}',
                        style: Theme.of(context).textTheme.bodySmall),
                    const SizedBox(height: 8),
                    Row(children: [
                      FilledButton(
                        onPressed: _busy
                            ? null
                            : () => _act('/assistance/requests/${r['id']}/accept', done: 'Accepted. Go to the driver.'),
                        child: const Text('Accept'),
                      ),
                      const SizedBox(width: 8),
                      OutlinedButton(
                        onPressed: _busy ? null : () => _act('/assistance/requests/${r['id']}/decline'),
                        child: const Text('Decline'),
                      ),
                    ]),
                  ]),
                ),
              ),
            const Text('Before you accept, only the approximate area (about 1 km) is shown.',
                style: TextStyle(fontSize: 12)),
          ]),
        if (payments.isNotEmpty)
          SectionCard(title: 'Payments', icon: Icons.payments, children: [
            for (final p in payments)
              ListTile(
                contentPadding: EdgeInsets.zero,
                title: Text('${p['assistance_id']} · ${p['driver_name']} · ${p['amount_label']}'),
                subtitle: Text((p['status'] as String).replaceAll('_', ' ')),
                trailing: p['status'] == 'PAYMENT_COMPLETED'
                    ? null
                    : PopupMenuButton<String>(
                        onSelected: (choice) => choice == 'received'
                            ? _act('/assistance/requests/${p['id']}/payment-received', done: 'Payment confirmed.')
                            : _act('/assistance/requests/${p['id']}/payment-problem',
                                body: {'note': 'Payment not received'}, done: 'Problem reported.'),
                        itemBuilder: (_) => const [
                          PopupMenuItem(value: 'received', child: Text('Payment received')),
                          PopupMenuItem(value: 'problem', child: Text('Report a payment problem')),
                        ],
                      ),
              ),
          ]),
        if (v['last_completed'] is Map)
          SectionCard(title: 'Last completed', icon: Icons.history, children: [
            InfoRow('Assistance', (v['last_completed'] as Map)['assistance_id'] as String?),
            InfoRow('Fare', (v['last_completed'] as Map)['amount_label'] as String?),
          ]),
      ]),
    );
  }

  Widget _activeCard(Map<String, dynamic> a) {
    final driver = a['driver'] as Map<String, dynamic>? ?? const {};
    final journey = a['journey'] as Map<String, dynamic>?;
    final status = a['status'] as String?;
    return SectionCard(
      title: 'Active assistance ${a['assistance_id']}',
      icon: Icons.directions_car,
      trailing: StatusChip(status),
      children: [
        InfoRow('Driver', driver['name'] as String?),
        InfoRow('Vehicle', [driver['vehicle_plate'], driver['vehicle']].whereType<String>().join(' · ')),
        InfoRow('Distance', a['distance_km'] != null ? '${a['distance_km']} km' : 'waiting for your location'),
        if (a['eta_min'] != null) InfoRow('ETA', '~${a['eta_min']} min'),
        if (journey != null) ...[
          InfoRow('Journey so far', '${journey['distance_so_far_km']} km'),
          InfoRow('Estimated fare', journey['estimated_fare_label'] as String?),
        ],
        const SizedBox(height: 8),
        Wrap(spacing: 8, runSpacing: 8, children: [
          OutlinedButton.icon(
            onPressed: () => callPhone(context, driver['phone'] as String?),
            icon: const Icon(Icons.call),
            label: const Text('Call driver'),
          ),
          if (status == 'ACCEPTED')
            FilledButton.icon(
              onPressed: _busy
                  ? null
                  : () => _act('/assistance/requests/${a['id']}/connect', done: 'Journey started.'),
              icon: const Icon(Icons.how_to_reg),
              label: const Text('I am with the driver'),
            ),
          if (status == 'DRIVER_CONNECTED')
            FilledButton.icon(
              onPressed: _busy
                  ? null
                  : () => _act('/assistance/requests/${a['id']}/complete',
                      done: 'Journey completed. The fare was calculated by SafeDrive.'),
              icon: const Icon(Icons.flag),
              label: const Text('Complete journey'),
            ),
        ]),
        const SizedBox(height: 4),
        Text(_locationError ?? 'Sharing your live location with this driver until the journey ends.',
            style: Theme.of(context).textTheme.bodySmall),
      ],
    );
  }
}
