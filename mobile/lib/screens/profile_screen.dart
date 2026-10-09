import 'package:flutter/material.dart';

import '../api/api_client.dart';
import '../models.dart';
import '../state/app_state.dart';
import '../widgets/common.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  late final TextEditingController _phone;

  @override
  void initState() {
    super.initState();
    _phone = TextEditingController(text: AppScope.read(context).user?.phone ?? '');
  }

  @override
  void dispose() {
    _phone.dispose();
    super.dispose();
  }

  Future<void> _savePhone() async {
    final state = AppScope.read(context);
    try {
      final body = await state.api.post('/me/phone', {'phone': _phone.text});
      _phone.text = body['phone'] as String;
      await state.refreshUser();
      if (mounted) showMessage(context, 'Phone number saved.');
    } on ApiException catch (e) {
      if (mounted) showMessage(context, e.message, error: true);
    }
  }

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    final user = state.user!;
    final checks = (user.badge?['checks'] as List?) ?? const [];
    final v = user.verification;
    final vehicle = user.driverProfile;
    return Scaffold(
      appBar: AppBar(title: const Text('My profile')),
      body: RefreshIndicator(
        onRefresh: state.refreshUser,
        child: ListView(padding: const EdgeInsets.all(16), children: [
          SectionCard(title: user.username, icon: Icons.person, children: [
            InfoRow('Role', user.roleLabel),
            InfoRow('Email', '${user.email}${user.emailVerified ? ' (verified)' : ' (not verified)'}'),
            InfoRow('Cooperative', user.cooperativeName),
            InfoRow('Group', user.membership?['group_name'] as String?),
            if (vehicle != null) ...[
              InfoRow('Vehicle plate', vehicle['vehicle_plate_number'] as String?),
              InfoRow('Vehicle', [vehicle['vehicle_make'], vehicle['vehicle_model']].whereType<String>().join(' ')),
            ],
          ]),
          if (v != null)
            SectionCard(
              title: 'Cooperative verification',
              icon: Icons.verified_user,
              trailing: StatusChip(v['status'] as String?),
              children: [
                InfoRow('Member code', v['member_code'] as String?),
                InfoRow('Manager', (v['manager'] as Map?)?['username'] as String?),
                if (v['note'] != null) InfoRow('Manager note', v['note'] as String?),
              ],
            ),
          if (checks.isNotEmpty)
            SectionCard(
              title: user.badgeVerified ? 'Verified by SafeDrive' : 'Verification checklist',
              icon: user.badgeVerified ? Icons.verified : Icons.checklist,
              children: [
                for (final c in checks)
                  Row(children: [
                    Icon((c as List)[1] == true ? Icons.check_circle : Icons.radio_button_unchecked,
                        size: 18, color: c[1] == true ? Colors.green : null),
                    const SizedBox(width: 8),
                    Expanded(child: Text('${c[0]}')),
                  ]),
                const SizedBox(height: 6),
                Text('The badge is a cooperative membership check, not a licence or a statement that a driver '
                    'is fit to drive.', style: Theme.of(context).textTheme.bodySmall),
              ],
            ),
          if (user.isDriver || user.isUmusare)
            SectionCard(title: 'Contact number', icon: Icons.phone, children: [
              TextField(
                controller: _phone,
                keyboardType: TextInputType.phone,
                decoration: const InputDecoration(hintText: '+250 78X XXX XXX'),
              ),
              const SizedBox(height: 8),
              FilledButton(onPressed: _savePhone, child: const Text('Save phone number')),
              if (user.isUmusare)
                Text('Drivers pay to this number after a journey.', style: Theme.of(context).textTheme.bodySmall),
            ]),
          Text('Server: ${state.apiUrl}  ·  ${statusLabel(user.role)}', style: Theme.of(context).textTheme.bodySmall),
        ]),
      ),
    );
  }
}
