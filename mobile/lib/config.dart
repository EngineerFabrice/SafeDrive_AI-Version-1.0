/// Build-time configuration. No secrets live in the app: it only knows the backend address.
///
/// The default backend URL is for the Android emulator, where 10.0.2.2 is the host computer.
/// For a physical phone, either pass the computer's LAN address at build time:
///
///     flutter run --dart-define=SAFEDRIVE_API_URL=http://192.168.1.20:5000
///
/// or change it in the app under "Server settings" (stored on the device).
library;

const String kDefaultApiUrl =
    String.fromEnvironment('SAFEDRIVE_API_URL', defaultValue: 'http://10.0.2.2:5000');

const String kApiPrefix = '/api/mobile/v1';

const String kAppName = 'SafeDrive AI';
const String kAppVersion = '1.0.0';
const String kAuthor = 'Fabrice NDAYISABA';
const String kAuthorEmail = 'fabricendayisaba16@gmail.com';

/// Shown wherever an AI result is displayed (mirrors the server's SYSTEM_NOTICE).
const String kAiDisclaimer =
    'SafeDrive AI uses a prototype camera model trained on a limited dataset. Its result is a visual '
    'pattern estimate, not a measurement of blood alcohol concentration and not proof of intoxication.';

/// Returns a normalised base URL (scheme + host [+ port], no trailing slash) or null when invalid.
String? normaliseBaseUrl(String raw) {
  final text = raw.trim().replaceAll(RegExp(r'/+$'), '');
  final uri = Uri.tryParse(text);
  if (uri == null || !(uri.scheme == 'http' || uri.scheme == 'https') || uri.host.isEmpty) {
    return null;
  }
  if (uri.hasQuery || uri.hasFragment) return null;
  return text;
}
