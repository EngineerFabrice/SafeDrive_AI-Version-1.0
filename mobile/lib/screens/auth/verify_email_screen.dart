import 'package:flutter/material.dart';

import '../../api/api_client.dart';
import '../../state/app_state.dart';
import '../../widgets/common.dart';

/// Enter the 6-digit code emailed by the server (after registration, or later while signed in).
class VerifyEmailScreen extends StatefulWidget {
  const VerifyEmailScreen({super.key});

  @override
  State<VerifyEmailScreen> createState() => _VerifyEmailScreenState();
}

class _VerifyEmailScreenState extends State<VerifyEmailScreen> {
  final _code = TextEditingController();
  bool _busy = false;
  String? _error;

  @override
  void dispose() {
    _code.dispose();
    super.dispose();
  }

  Future<void> _verify() async {
    final code = _code.text.replaceAll(RegExp(r'\D'), '');
    if (code.length != 6) {
      setState(() => _error = 'Enter the 6-digit code from the email.');
      return;
    }
    setState(() {
      _busy = true;
      _error = null;
    });
    final state = AppScope.read(context);
    final wasSignedIn = state.user != null;
    try {
      await state.verifyEmail(code);
      if (!mounted) return;
      showMessage(context, 'Email verified successfully.');
      if (wasSignedIn) {
        Navigator.pop(context);
      } else {
        Navigator.popUntil(context, (r) => r.isFirst);
      }
    } on ApiException catch (e) {
      if (mounted) setState(() => _error = e.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _resend() async {
    try {
      final message = await AppScope.read(context).resendCode();
      if (mounted) showMessage(context, message);
    } on ApiException catch (e) {
      if (mounted) showMessage(context, e.message, error: true);
    }
  }

  @override
  Widget build(BuildContext context) {
    final state = AppScope.of(context);
    final pending = state.pending;
    final intro = pending != null
        ? (pending.message.isNotEmpty ? pending.message : 'We sent a code to ${pending.maskedEmail}.')
        : 'Request a code, then enter it below to verify ${state.user?.email ?? 'your email address'}.';
    return Scaffold(
      appBar: AppBar(title: const Text('Verify your email')),
      body: ListView(padding: const EdgeInsets.all(24), children: [
        const Icon(Icons.mark_email_read, size: 56),
        const SizedBox(height: 12),
        Text(intro, textAlign: TextAlign.center),
        const SizedBox(height: 16),
        TextField(
          key: const Key('otp-code'),
          controller: _code,
          keyboardType: TextInputType.number,
          maxLength: 6,
          textAlign: TextAlign.center,
          style: const TextStyle(fontSize: 24, letterSpacing: 8),
          decoration: const InputDecoration(labelText: '6-digit code', counterText: ''),
          onSubmitted: (_) => _verify(),
        ),
        if (_error != null)
          Text(_error!, textAlign: TextAlign.center, style: TextStyle(color: Theme.of(context).colorScheme.error)),
        const SizedBox(height: 16),
        FilledButton(onPressed: _busy ? null : _verify, child: const Text('Verify')),
        TextButton(onPressed: _resend, child: const Text('Resend code')),
        const SizedBox(height: 8),
        Text('Codes expire after 10 minutes. Check your spam folder if the email does not arrive.',
            textAlign: TextAlign.center, style: Theme.of(context).textTheme.bodySmall),
      ]),
    );
  }
}
