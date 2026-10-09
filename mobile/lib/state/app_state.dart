import 'package:flutter/widgets.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

import '../api/api_client.dart';
import '../config.dart';
import '../models.dart';

/// Where the app keeps the server address (plain preferences) and the sign-in token (secure storage).
abstract class SettingsStore {
  Future<String?> readToken();
  Future<void> writeToken(String? token);
  Future<String?> readApiUrl();
  Future<void> writeApiUrl(String url);
}

class DeviceSettingsStore implements SettingsStore {
  static const _tokenKey = 'safedrive_token';
  static const _urlKey = 'safedrive_api_url';
  final _secure = const FlutterSecureStorage();

  @override
  Future<String?> readToken() => _secure.read(key: _tokenKey);

  @override
  Future<void> writeToken(String? token) =>
      token == null ? _secure.delete(key: _tokenKey) : _secure.write(key: _tokenKey, value: token);

  @override
  Future<String?> readApiUrl() async => (await SharedPreferences.getInstance()).getString(_urlKey);

  @override
  Future<void> writeApiUrl(String url) async => (await SharedPreferences.getInstance()).setString(_urlKey, url);
}

class MemorySettingsStore implements SettingsStore {
  MemorySettingsStore({this.token, this.apiUrl});

  String? token;
  String? apiUrl;

  @override
  Future<String?> readToken() async => token;

  @override
  Future<void> writeToken(String? value) async => token = value;

  @override
  Future<String?> readApiUrl() async => apiUrl;

  @override
  Future<void> writeApiUrl(String url) async => apiUrl = url;
}

/// Pending email verification right after registration (opaque, encrypted by the server).
class PendingVerification {
  PendingVerification(this.token, this.maskedEmail, this.message);

  String token;
  final String maskedEmail;
  final String message;
}

/// Session state: server address, token, signed-in user. Screens read it through [AppScope].
class AppState extends ChangeNotifier {
  AppState({SettingsStore? store, http.Client? httpClient})
      : _store = store ?? DeviceSettingsStore(),
        api = ApiClient(baseUrl: kDefaultApiUrl, client: httpClient) {
    api.onUnauthorized = _expired;
  }

  final SettingsStore _store;
  final ApiClient api;

  bool initializing = true;
  AppUser? user;
  PendingVerification? pending;
  String? notice;      // one-off message for the sign-in screen (e.g. "session expired")

  String get apiUrl => api.baseUrl;

  Future<void> init() async {
    final saved = await _store.readApiUrl();
    if (saved != null && normaliseBaseUrl(saved) != null) api.baseUrl = saved;
    api.token = await _store.readToken();
    if (api.token != null) {
      try {
        await refreshUser();
      } on ApiException catch (e) {
        if (e.isUnauthorized) {
          await _clearToken();
        } else {
          notice = e.message;     // keep the token: the server may just be unreachable right now
          await _clearToken(keepStored: true);
        }
      }
    }
    initializing = false;
    notifyListeners();
  }

  Future<void> setApiUrl(String url) async {
    final normalised = normaliseBaseUrl(url);
    if (normalised == null) throw ArgumentError('Enter an address like http://192.168.1.20:5000');
    api.baseUrl = normalised;
    await _store.writeApiUrl(normalised);
    notifyListeners();
  }

  Future<void> login(String email, String password) async {
    final body = await api.post('/auth/login', {'email': email.trim(), 'password': password, 'device_name': 'Android'});
    await _signedIn(body);
  }

  Future<void> _signedIn(Map<String, dynamic> body) async {
    api.token = body['token'] as String;
    await _store.writeToken(api.token);
    user = AppUser(body['user'] as Map<String, dynamic>);
    pending = null;
    notice = null;
    notifyListeners();
  }

  Future<Map<String, dynamic>> register(Map<String, dynamic> form) async {
    final body = await api.post('/auth/register', form);
    pending = PendingVerification(body['pending_token'] as String, body['masked_email'] as String? ?? '',
        body['message'] as String? ?? '');
    notifyListeners();
    return body;
  }

  /// Verify the emailed code for a pending registration (signs in) or for the signed-in user.
  Future<void> verifyEmail(String code) async {
    if (user != null) {
      final body = await api.post('/auth/verify-email', {'code': code});
      user = AppUser(body['user'] as Map<String, dynamic>);
      notifyListeners();
      return;
    }
    final p = pending;
    if (p == null) throw ApiException(400, 'Sign in to verify your email address.');
    try {
      final body = await api.post('/auth/verify-email', {'pending_token': p.token, 'code': code});
      if (body['token'] != null) {
        await _signedIn(body);
      } else {
        pending = null;
        notice = 'Email verified. Please sign in.';
        notifyListeners();
      }
    } on ApiException catch (e) {
      final next = e.body?['pending_token'];
      if (next is String) p.token = next;
      rethrow;
    }
  }

  Future<String> resendCode() async {
    if (user != null) {
      return (await api.post('/auth/resend-code'))['message'] as String? ?? 'Code sent.';
    }
    final p = pending;
    if (p == null) throw ApiException(400, 'Sign in to verify your email address.');
    try {
      final body = await api.post('/auth/resend-code', {'pending_token': p.token});
      p.token = body['pending_token'] as String? ?? p.token;
      return body['message'] as String? ?? 'Code sent.';
    } on ApiException catch (e) {
      final next = e.body?['pending_token'];
      if (next is String) p.token = next;
      rethrow;
    }
  }

  void cancelPending() {
    pending = null;
    notifyListeners();
  }

  Future<void> refreshUser() async {
    final body = await api.get('/me');
    user = AppUser(body['user'] as Map<String, dynamic>);
    notifyListeners();
  }

  void updateUser(Map<String, dynamic> raw) {
    user = AppUser(raw);
    notifyListeners();
  }

  Future<void> logout() async {
    try {
      if (api.token != null) await api.post('/auth/logout');
    } on ApiException {
      // signing out locally always works, even offline
    }
    await _clearToken();
    notifyListeners();
  }

  Future<void> _clearToken({bool keepStored = false}) async {
    api.token = null;
    user = null;
    if (!keepStored) await _store.writeToken(null);
  }

  void _expired() {
    if (user == null) return;
    notice = 'Your session has ended. Please sign in again.';
    _clearToken().then((_) => notifyListeners());
  }
}

/// Makes [AppState] available to every screen and rebuilds dependants when it changes.
class AppScope extends InheritedNotifier<AppState> {
  const AppScope({super.key, required AppState state, required super.child}) : super(notifier: state);

  static AppState of(BuildContext context) {
    final scope = context.dependOnInheritedWidgetOfExactType<AppScope>();
    assert(scope != null, 'AppScope missing');
    return scope!.notifier!;
  }

  /// Read without subscribing to changes (for callbacks).
  static AppState read(BuildContext context) =>
      context.getInheritedWidgetOfExactType<AppScope>()!.notifier!;
}
