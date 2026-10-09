import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import '../../api/api_client.dart';
import '../../state/app_state.dart';
import '../../widgets/common.dart';
import 'verify_email_screen.dart';

/// Driver / Umusare self-registration (managers and administrators are appointed on the web).
class RegisterScreen extends StatefulWidget {
  const RegisterScreen({super.key});

  @override
  State<RegisterScreen> createState() => _RegisterScreenState();
}

class _RegisterScreenState extends State<RegisterScreen> {
  final _form = GlobalKey<FormState>();
  final _name = TextEditingController();
  final _email = TextEditingController();
  final _phone = TextEditingController();
  final _password = TextEditingController();
  final _confirm = TextEditingController();
  final _plate = TextEditingController();
  final _make = TextEditingController();
  final _model = TextEditingController();

  Map<String, dynamic>? _options;
  String? _loadError;
  String _role = 'driver';
  int? _coopId;
  int? _groupId;
  String? _vehicleType;
  bool _terms = false;
  bool _busy = false;
  List<String> _errors = const [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    for (final c in [_name, _email, _phone, _password, _confirm, _plate, _make, _model]) {
      c.dispose();
    }
    super.dispose();
  }

  Future<void> _load() async {
    setState(() => _loadError = null);
    try {
      final options = await AppScope.read(context).api.get('/registration-options');
      if (mounted) setState(() => _options = options);
    } on ApiException catch (e) {
      if (mounted) setState(() => _loadError = e.message);
    }
  }

  List<Map<String, dynamic>> _list(String key) =>
      ((_options?[key] as List?) ?? const []).cast<Map<String, dynamic>>();

  Future<void> _submit() async {
    if (!_form.currentState!.validate()) return;
    if (!_terms) {
      setState(() => _errors = ['Please read and accept the Terms & Conditions and Privacy Policy.']);
      return;
    }
    setState(() {
      _busy = true;
      _errors = const [];
    });
    final state = AppScope.read(context);
    try {
      await state.register({
        'username': _name.text.trim(),
        'email': _email.text.trim(),
        'phone': _phone.text.trim(),
        'password': _password.text,
        'confirm_password': _confirm.text,
        'role': _role,
        'cooperative_id': _coopId,
        'group_id': _groupId,
        'accept_terms': _terms,
        if (_role == 'driver') ...{
          'vehicle_plate_number': _plate.text.trim(),
          'vehicle_make': _make.text.trim(),
          'vehicle_model': _model.text.trim(),
          'vehicle_type': _vehicleType ?? '',
        },
      });
      if (mounted) {
        Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => const VerifyEmailScreen()));
      }
    } on ApiException catch (e) {
      final errors = (e.body?['errors'] as List?)?.cast<String>();
      if (mounted) setState(() => _errors = errors ?? [e.message]);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    if (_options == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Create account')),
        body: _loadError != null
            ? ErrorRetry(message: _loadError!, onRetry: _load)
            : const Center(child: CircularProgressIndicator()),
      );
    }
    final coops = _list('cooperatives');
    final groups = _list('groups').where((g) => g['cooperative_id'] == _coopId).toList();
    final types = _list('vehicle_types');
    return Scaffold(
      appBar: AppBar(title: const Text('Create account')),
      body: Form(
        key: _form,
        child: ListView(padding: const EdgeInsets.all(16), children: [
          SegmentedButton<String>(
            segments: const [
              ButtonSegment(value: 'driver', label: Text('Driver'), icon: Icon(Icons.directions_car)),
              ButtonSegment(value: 'umusare', label: Text('Umusare'), icon: Icon(Icons.support)),
            ],
            selected: {_role},
            onSelectionChanged: (s) => setState(() => _role = s.first),
          ),
          const SizedBox(height: 12),
          TextFormField(
            controller: _name,
            decoration: const InputDecoration(labelText: 'Full name'),
            validator: (v) => (v == null || v.trim().length < 3) ? 'Enter your name (3–60 characters)' : null,
          ),
          TextFormField(
            controller: _email,
            keyboardType: TextInputType.emailAddress,
            decoration: const InputDecoration(labelText: 'Email'),
            validator: (v) => (v == null || !v.contains('@')) ? 'Enter a valid email address' : null,
          ),
          TextFormField(
            controller: _phone,
            keyboardType: TextInputType.phone,
            decoration: InputDecoration(
                labelText: _role == 'umusare' ? 'Phone (used for payments)' : 'Phone (optional)',
                hintText: '+250 78X XXX XXX'),
          ),
          const SizedBox(height: 8),
          DropdownButtonFormField<int>(
            initialValue: _coopId,
            isExpanded: true,
            decoration: const InputDecoration(labelText: 'Cooperative'),
            items: [
              for (final c in coops)
                DropdownMenuItem(value: c['id'] as int, child: Text('${c['name']} (${c['code']})')),
            ],
            onChanged: (v) => setState(() {
              _coopId = v;
              _groupId = null;
            }),
            validator: (v) => v == null ? 'Choose your cooperative' : null,
          ),
          if (groups.isNotEmpty)
            DropdownButtonFormField<int?>(
              initialValue: _groupId,
              isExpanded: true,
              decoration: const InputDecoration(labelText: 'Group (optional)'),
              items: [
                const DropdownMenuItem<int?>(value: null, child: Text('No group')),
                for (final g in groups) DropdownMenuItem<int?>(value: g['id'] as int, child: Text('${g['name']}')),
              ],
              onChanged: (v) => setState(() => _groupId = v),
            ),
          if (_role == 'driver') ...[
            const SizedBox(height: 8),
            Text('Vehicle (the plate is required before your manager can verify you)',
                style: Theme.of(context).textTheme.bodySmall),
            TextFormField(controller: _plate, decoration: const InputDecoration(labelText: 'Plate number')),
            TextFormField(controller: _make, decoration: const InputDecoration(labelText: 'Make (optional)')),
            TextFormField(controller: _model, decoration: const InputDecoration(labelText: 'Model (optional)')),
            DropdownButtonFormField<String?>(
              initialValue: _vehicleType,
              isExpanded: true,
              decoration: const InputDecoration(labelText: 'Vehicle type (optional)'),
              items: [
                const DropdownMenuItem<String?>(value: null, child: Text('—')),
                for (final t in types)
                  DropdownMenuItem<String?>(value: t['value'] as String, child: Text('${t['label']}')),
              ],
              onChanged: (v) => setState(() => _vehicleType = v),
            ),
          ],
          TextFormField(
            controller: _password,
            obscureText: true,
            decoration: const InputDecoration(labelText: 'Password (8+ characters, letters and numbers)'),
            validator: (v) => (v == null || v.length < 8) ? 'At least 8 characters' : null,
          ),
          TextFormField(
            controller: _confirm,
            obscureText: true,
            decoration: const InputDecoration(labelText: 'Confirm password'),
            validator: (v) => v != _password.text ? 'Passwords do not match' : null,
          ),
          const SizedBox(height: 8),
          CheckboxListTile(
            value: _terms,
            contentPadding: EdgeInsets.zero,
            controlAffinity: ListTileControlAffinity.leading,
            onChanged: (v) => setState(() => _terms = v ?? false),
            title: const Text('I have read and accept the Terms & Conditions and the Privacy Policy'),
          ),
          Wrap(spacing: 8, children: [
            TextButton(
                onPressed: () => launchUrl(Uri.parse('${state.apiUrl}/terms'), mode: LaunchMode.externalApplication),
                child: const Text('Read Terms')),
            TextButton(
                onPressed: () =>
                    launchUrl(Uri.parse('${state.apiUrl}/privacy'), mode: LaunchMode.externalApplication),
                child: const Text('Read Privacy Policy')),
          ]),
          for (final e in _errors)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(e, style: TextStyle(color: Theme.of(context).colorScheme.error)),
            ),
          const SizedBox(height: 12),
          FilledButton(
            onPressed: _busy ? null : _submit,
            child: _busy
                ? const SizedBox(height: 20, width: 20, child: CircularProgressIndicator(strokeWidth: 2))
                : const Text('Create account'),
          ),
          const SizedBox(height: 8),
          Text('After email verification, your cooperative manager verifies your account.',
              style: Theme.of(context).textTheme.bodySmall),
        ]),
      ),
    );
  }
}
