import 'dart:async';

import 'package:flutter/material.dart';

import '../../api/api_client.dart';
import '../../services/location_service.dart';
import '../../state/app_state.dart';
import '../../widgets/common.dart';

const _active = ['REQUESTED', 'MATCHING', 'ACCEPTED', 'DRIVER_CONNECTED'];
const _sharing = ['ACCEPTED', 'DRIVER_CONNECTED'];

/// Driver side of the assistance workflow: request, matching, accepted Umusare, journey, payment, rating.
/// The server decides every state change; this screen polls it and sends button presses.
class AssistanceScreen extends StatefulWidget {
  const AssistanceScreen({super.key, this.aiAlert = false});

  /// Opened from a POTENTIALLY_NOT_SOBER alert: the request claims AI_TRIGGERED, which the server
  /// accepts only if this driver's own monitoring session really shows that assessment.
  final bool aiAlert;

  @override
  State<AssistanceScreen> createState() => _AssistanceScreenState();
}

class _AssistanceScreenState extends State<AssistanceScreen> {
  Map<String, dynamic>? _req;
  bool _loaded = false;
  bool _busy = false;
  String? _error;
  String? _sharingError;
  Timer? _poll;
  Timer? _share;

  ApiClient get _api => AppScope.read(context).api;

  @override
  void initState() {
    super.initState();
    _refresh();
    _poll = Timer.periodic(const Duration(seconds: 5), (_) => _refresh());
  }

  @override
  void dispose() {
    _poll?.cancel();
    _share?.cancel();
    super.dispose();
  }

  Future<void> _refresh() async {
    try {
      final body = await _api.get('/assistance/requests/current');
      if (!mounted) return;
      setState(() {
        _req = body['request'] as Map<String, dynamic>?;
        _loaded = true;
        _error = null;
      });
      _updateSharing();
    } on ApiException catch (e) {
      if (mounted) setState(() => _error = e.message);
    }
  }

  /// Live location only while an Umusare is assigned (the server refuses it in any other state).
  void _updateSharing() {
    final sharing = _sharing.contains(_req?['status']);
    if (sharing && _share == null) {
      _sendLocation();
      _share = Timer.periodic(const Duration(seconds: 15), (_) => _sendLocation());
    } else if (!sharing && _share != null) {
      _share!.cancel();
      _share = null;
    }
  }

  Future<void> _sendLocation() async {
    final id = _req?['id'];
    if (id == null) return;
    try {
      final fix = await LocationService.current();
      await _api.post('/assistance/requests/$id/location', fix.toJson());
      if (mounted && _sharingError != null) setState(() => _sharingError = null);
    } on LocationException catch (e) {
      if (mounted) setState(() => _sharingError = e.message);
    } on ApiException catch (e) {
      if (e.code != 'SHARING_STOPPED' && mounted) setState(() => _sharingError = e.message);
    }
  }

  Future<void> _act(Future<Map<String, dynamic>> Function() call, {String? done}) async {
    setState(() => _busy = true);
    try {
      final body = await call();
      if (!mounted) return;
      if (body['request'] is Map) setState(() => _req = body['request'] as Map<String, dynamic>);
      if (done != null) showMessage(context, done);
      _updateSharing();
    } on ApiException catch (e) {
      if (mounted) showMessage(context, e.message, error: true);
    } on LocationException catch (e) {
      if (mounted) showMessage(context, e.message, error: true);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _request() async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Request an Umusare?'),
        content: const Text('Your current location will be sent to SafeDrive to find a verified Umusare near '
            'you. Before one accepts, Umusare only see your approximate area (about 1 km). Your exact '
            'position is shared only with the Umusare who accepts, until the journey ends.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Share location & request')),
        ],
      ),
    );
    if (ok != true) return;
    await _act(() async {
      final fix = await LocationService.current();
      return _api.post('/assistance/requests',
          {...fix.toJson(), 'trigger': widget.aiAlert ? 'AI_TRIGGERED' : 'DRIVER_INITIATED'});
    }, done: 'Request sent. Looking for an Umusare…');
  }

  Future<String?> _askText(String title, String hint) {
    final controller = TextEditingController();
    return showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(title),
        content: TextField(controller: controller, maxLines: 3, maxLength: 500,
            decoration: InputDecoration(hintText: hint)),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(ctx, controller.text), child: const Text('Send')),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Umusare assistance')),
      body: !_loaded
          ? (_error != null
              ? ErrorRetry(message: _error!, onRetry: _refresh)
              : const Center(child: CircularProgressIndicator()))
          : RefreshIndicator(onRefresh: _refresh, child: _content()),
    );
  }

  Widget _content() {
    final req = _req;
    final status = req?['status'] as String?;
    final children = <Widget>[
      if (_error != null) Text(_error!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
    ];
    if (req == null || !_active.contains(status)) {
      if (req != null) children.add(_finished(req));
      children.add(SectionCard(title: 'Request a safe driver', icon: Icons.sos, children: [
        if (widget.aiAlert)
          const Padding(
            padding: EdgeInsets.only(bottom: 8),
            child: Text('Your monitoring session reported POTENTIALLY NOT SOBER. Please do not drive.',
                style: TextStyle(fontWeight: FontWeight.bold)),
          ),
        const Text('A verified Umusare from any cooperative near you will be asked to come and drive you.'),
        const SizedBox(height: 8),
        FilledButton.icon(
          key: const Key('request-assistance'),
          style: FilledButton.styleFrom(backgroundColor: Colors.red[700]),
          onPressed: _busy ? null : _request,
          icon: const Icon(Icons.sos),
          label: const Text('Request assistance now'),
        ),
      ]));
      return ListView(padding: const EdgeInsets.all(16), children: children);
    }

    final id = req['id'];
    children.add(SectionCard(
      title: 'Assistance ${req['assistance_id']}',
      icon: Icons.support,
      trailing: StatusChip(status),
      children: [
        InfoRow('Type', req['type_label'] as String?),
        if (status == 'MATCHING' || status == 'REQUESTED') ...[
          InfoRow('Search radius', '${req['search_radius_km'] ?? '—'} km (round ${req['matching_round'] ?? 1})'),
          InfoRow('Umusare asked', '${req['umusare_contacted'] ?? 0}'),
          const SizedBox(height: 8),
          const LinearProgressIndicator(),
          const SizedBox(height: 8),
          const Text('Waiting for a nearby Umusare to accept…'),
        ],
        if (req['pricing'] is Map) InfoRow('Rate', (req['pricing'] as Map)['rate_label'] as String?),
      ],
    ));
    final umusare = req['umusare'] as Map<String, dynamic>?;
    if (umusare != null) {
      final profile = umusare['profile'] as Map<String, dynamic>?;
      children.add(SectionCard(title: 'Your Umusare', icon: Icons.person_pin_circle, children: [
        InfoRow('Name', umusare['name'] as String?),
        InfoRow('Umusare ID', profile?['umusare_id'] as String?),
        InfoRow('Cooperative', umusare['cooperative'] as String?),
        InfoRow('Verified', (profile?['verification'] as Map?)?['eligible'] == true ? 'Yes, by cooperative' : '—'),
        InfoRow('Distance', umusare['distance_km'] != null ? '${umusare['distance_km']} km' : 'location pending'),
        if (umusare['eta_min'] != null) InfoRow('Arrives in', '~${umusare['eta_min']} min'),
        const SizedBox(height: 8),
        OutlinedButton.icon(
          onPressed: () => callPhone(context, umusare['phone'] as String?),
          icon: const Icon(Icons.call),
          label: Text('Call ${umusare['phone'] ?? ''}'),
        ),
      ]));
    }
    final journey = req['journey'] as Map<String, dynamic>?;
    if (journey != null) {
      children.add(SectionCard(title: 'Journey in progress', icon: Icons.route, children: [
        InfoRow('Distance so far', '${journey['distance_so_far_km']} km'),
        InfoRow('Estimated fare', journey['estimated_fare_label'] as String?),
        const Text('The final fare is calculated by SafeDrive when the Umusare completes the journey.'),
      ]));
    }
    if (_sharing.contains(status)) {
      children.add(ListTile(
        leading: const Icon(Icons.my_location),
        title: const Text('Sharing your live location with your Umusare'),
        subtitle: Text(_sharingError ?? 'Stops automatically when the assistance ends.'),
      ));
    }
    if (status != 'DRIVER_CONNECTED') {
      children.add(OutlinedButton.icon(
        onPressed: _busy ? null : () => _act(() => _api.post('/assistance/requests/$id/cancel'), done: 'Cancelled.'),
        icon: const Icon(Icons.close),
        label: const Text('Cancel request'),
      ));
    }
    return ListView(padding: const EdgeInsets.all(16), children: children);
  }

  /// The latest finished request: payment, rating, problem report, or the fallback contacts.
  Widget _finished(Map<String, dynamic> req) {
    final id = req['id'];
    final status = req['status'] as String?;
    final payment = req['payment'] as Map<String, dynamic>?;
    final summary = req['summary'] as Map<String, dynamic>?;
    final contacts = (req['fallback_contacts'] as List?)?.cast<Map<String, dynamic>>() ?? const [];
    return SectionCard(
      title: 'Last assistance ${req['assistance_id']}',
      icon: Icons.history,
      trailing: StatusChip(status),
      children: [
        if (status == 'NO_UMUSARE_AVAILABLE') ...[
          const Text('No Umusare was available. Contact your cooperative manager:'),
          for (final c in contacts)
            ListTile(
              contentPadding: EdgeInsets.zero,
              title: Text('${c['name']}'),
              subtitle: Text('${c['phone'] ?? 'no phone'}'),
              trailing: IconButton(icon: const Icon(Icons.call), onPressed: () => callPhone(context, c['phone'])),
            ),
        ],
        if (payment != null) ...[
          InfoRow('Distance', '${payment['final_distance_km'] ?? '—'} km'),
          InfoRow('Fare', payment['amount_label'] as String?),
          InfoRow('Pay to', '${payment['pay_to_phone'] ?? '—'} (${(payment['payee'] as Map?)?['name'] ?? ''})'),
          InfoRow('Payment', (payment['status'] as String?)?.replaceAll('_', ' ')),
          if (payment['status'] == 'PAYMENT_PENDING' || payment['status'] == 'PAYMENT_DISPUTED')
            FilledButton(
              onPressed: _busy
                  ? null
                  : () => _act(() => _api.post('/assistance/requests/$id/payment-sent'),
                      done: 'Marked as sent. The Umusare will confirm.'),
              child: const Text('I have sent the payment'),
            ),
          if (payment['status'] == 'PAYMENT_COMPLETED' && summary?['rating'] == null) ...[
            const SizedBox(height: 8),
            const Text('Rate your Umusare:'),
            Row(children: [
              for (var star = 1; star <= 5; star++)
                IconButton(
                  icon: const Icon(Icons.star_border),
                  onPressed: () => _act(() => _api.post('/assistance/requests/$id/rate', {'rating': star}),
                      done: 'Thank you for your rating.'),
                ),
            ]),
          ],
          if (summary?['rating'] != null) InfoRow('Your rating', '${summary!['rating']} / 5'),
          if (summary?['problem_reported'] != true)
            TextButton(
              onPressed: () async {
                final text = await _askText('Report a problem', 'What went wrong?');
                if (text != null && text.trim().isNotEmpty) {
                  await _act(() => _api.post('/assistance/requests/$id/report', {'text': text}),
                      done: 'Problem reported to SafeDrive.');
                }
              },
              child: const Text('Report a problem'),
            ),
        ],
      ],
    );
  }
}
