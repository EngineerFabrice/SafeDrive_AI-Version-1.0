import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import '../config.dart';
import '../models.dart';

void showMessage(BuildContext context, String message, {bool error = false}) {
  ScaffoldMessenger.of(context)
    ..hideCurrentSnackBar()
    ..showSnackBar(SnackBar(
      content: Text(message),
      backgroundColor: error ? Theme.of(context).colorScheme.error : null,
    ));
}

Future<void> callPhone(BuildContext context, String? phone) async {
  if (phone == null || phone.isEmpty) {
    showMessage(context, 'No phone number available.', error: true);
    return;
  }
  final ok = await launchUrl(Uri(scheme: 'tel', path: phone));
  if (!ok && context.mounted) showMessage(context, 'Could not open the dialer for $phone.', error: true);
}

class SectionCard extends StatelessWidget {
  const SectionCard({super.key, required this.title, this.icon, required this.children, this.trailing});

  final String title;
  final IconData? icon;
  final List<Widget> children;
  final Widget? trailing;

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 6),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(children: [
              if (icon != null) ...[Icon(icon, color: Theme.of(context).colorScheme.primary), const SizedBox(width: 8)],
              Expanded(child: Text(title, style: Theme.of(context).textTheme.titleMedium)),
              ?trailing,
            ]),
            const SizedBox(height: 8),
            ...children,
          ],
        ),
      ),
    );
  }
}

class InfoRow extends StatelessWidget {
  const InfoRow(this.label, this.value, {super.key});

  final String label;
  final String? value;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        SizedBox(width: 130, child: Text(label, style: TextStyle(color: Theme.of(context).hintColor))),
        Expanded(child: Text(value == null || value!.isEmpty ? '—' : value!)),
      ]),
    );
  }
}

class StatusChip extends StatelessWidget {
  const StatusChip(this.status, {super.key});

  final String? status;

  Color _color() {
    switch (status) {
      case 'VERIFIED':
      case 'AVAILABLE':
      case 'COMPLETED':
      case 'PAYMENT_COMPLETED':
      case 'ACCEPTED':
      case 'DRIVER_CONNECTED':
      case 'APPROVED':
        return const Color(0xFF2E7D32);
      case 'REJECTED':
      case 'SUSPENDED':
      case 'CANCELLED':
      case 'NO_UMUSARE_AVAILABLE':
      case 'PAYMENT_DISPUTED':
        return const Color(0xFFC62828);
      case 'OFFLINE':
        return const Color(0xFF757575);
      default:
        return const Color(0xFF1565C0);
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = _color();
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(color: c.withValues(alpha: 0.12), borderRadius: BorderRadius.circular(12)),
      child: Text(statusLabel(status), style: TextStyle(color: c, fontWeight: FontWeight.w600, fontSize: 12)),
    );
  }
}

class Disclaimer extends StatelessWidget {
  const Disclaimer({super.key, this.text = kAiDisclaimer});

  final String text;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        const Icon(Icons.info_outline, size: 18),
        const SizedBox(width: 8),
        Expanded(child: Text(text, style: Theme.of(context).textTheme.bodySmall)),
      ]),
    );
  }
}

/// The temporal assessment (SOBER / UNCERTAIN / POTENTIALLY_NOT_SOBER / ASSESSING) from the server.
class AssessmentCard extends StatelessWidget {
  const AssessmentCard({super.key, required this.assessment});

  final Map<String, dynamic>? assessment;

  @override
  Widget build(BuildContext context) {
    final label = assessment?['assessment'] as String?;
    final style = AssessmentStyle.of(label);
    final reasons = (assessment?['reasons'] as List?)?.cast<String>() ?? const [];
    final valid = assessment?['valid_frames'];
    final total = assessment?['total_frames'];
    final confidence = assessment?['confidence'];
    return Container(
      key: const Key('assessment-card'),
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: style.color.withValues(alpha: 0.10),
        border: Border.all(color: style.color, width: 2),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Icon(style.icon, color: style.color, size: 32),
          const SizedBox(width: 12),
          Expanded(
            child: Text(style.title,
                style: Theme.of(context).textTheme.titleLarge?.copyWith(color: style.color, fontWeight: FontWeight.bold)),
          ),
        ]),
        const SizedBox(height: 6),
        Text(style.description),
        if (reasons.isNotEmpty) Text('Reason: ${reasons.map(reasonLabel).join(', ')}'),
        if (valid != null)
          Text('Usable frames: $valid / $total'
              '${confidence is num ? ' · confidence ${(confidence * 100).round()}%' : ''}',
              style: Theme.of(context).textTheme.bodySmall),
      ]),
    );
  }
}

class ErrorRetry extends StatelessWidget {
  const ErrorRetry({super.key, required this.message, required this.onRetry});

  final String message;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          const Icon(Icons.cloud_off, size: 48),
          const SizedBox(height: 12),
          Text(message, textAlign: TextAlign.center),
          const SizedBox(height: 12),
          FilledButton.icon(onPressed: onRetry, icon: const Icon(Icons.refresh), label: const Text('Retry')),
        ]),
      ),
    );
  }
}
