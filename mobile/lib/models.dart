import 'package:flutter/material.dart';

/// The signed-in user as returned by GET /me (role, membership, verification, badge).
class AppUser {
  AppUser(this.raw);

  final Map<String, dynamic> raw;

  int get id => raw['id'] as int;
  String get username => raw['username'] as String? ?? '';
  String get email => raw['email'] as String? ?? '';
  String get role => raw['role'] as String? ?? 'driver';
  String? get phone => raw['phone'] as String?;
  bool get emailVerified => raw['email_verified'] == true;
  bool get needsTerms => raw['needs_terms_acceptance'] == true;
  Map<String, dynamic>? get membership => raw['membership'] as Map<String, dynamic>?;
  Map<String, dynamic>? get verification => raw['verification'] as Map<String, dynamic>?;
  Map<String, dynamic>? get badge => raw['badge'] as Map<String, dynamic>?;
  Map<String, dynamic>? get driverProfile => raw['driver_profile'] as Map<String, dynamic>?;
  Map<String, dynamic>? get umusareProfile => raw['umusare_profile'] as Map<String, dynamic>?;

  bool get isDriver => role == 'driver';
  bool get isUmusare => role == 'umusare';
  bool get isManager => role == 'manager';
  bool get isAdmin => role == 'admin';
  bool get canMonitor => isDriver || isAdmin;

  String get cooperativeName => membership?['cooperative_name'] as String? ?? '—';
  String get verificationStatus => verification?['status'] as String? ?? 'PENDING';
  bool get badgeVerified => badge?['verified'] == true;

  String get roleLabel => const {
        'driver': 'Driver',
        'umusare': 'Umusare',
        'manager': 'Cooperative manager',
        'admin': 'Administrator',
      }[role] ??
      role;
}

/// How the app presents the temporal assessment labels produced by the backend.
class AssessmentStyle {
  const AssessmentStyle(this.title, this.description, this.color, this.icon);

  final String title;
  final String description;
  final Color color;
  final IconData icon;

  static AssessmentStyle of(String? label) {
    switch (label) {
      case 'SOBER':
        return const AssessmentStyle('SOBER', 'No alcohol-related visual pattern detected in recent frames.',
            Color(0xFF2E7D32), Icons.check_circle);
      case 'UNCERTAIN':
        return const AssessmentStyle('UNCERTAIN', 'Not enough reliable evidence for a result.',
            Color(0xFFF9A825), Icons.help);
      case 'POTENTIALLY_NOT_SOBER':
        return const AssessmentStyle('POTENTIALLY NOT SOBER',
            'Recent frames show patterns the prototype model associates with alcohol. Do not drive; '
                'request assistance.',
            Color(0xFFC62828), Icons.warning);
      case 'ASSESSING':
        return const AssessmentStyle('ASSESSING', 'Collecting frames for an initial assessment.',
            Color(0xFF1565C0), Icons.hourglass_top);
      default:
        return const AssessmentStyle('NO RESULT', 'No assessment yet.', Color(0xFF757575), Icons.remove_circle);
    }
  }
}

/// Human-readable names for the reasons attached to an UNCERTAIN result.
String reasonLabel(String reason) => const {
      'insufficient_valid_frames': 'not enough usable face frames',
      'too_many_invalid_frames': 'face often missing or low quality',
      'low_model_confidence': 'model not confident',
      'ambiguous_temporal_score': 'score between thresholds',
    }[reason] ??
    reason;

String statusLabel(String? status) => (status ?? '—').replaceAll('_', ' ');
