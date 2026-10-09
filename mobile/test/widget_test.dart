import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:safedrive_mobile/api/api_client.dart';
import 'package:safedrive_mobile/config.dart';
import 'package:safedrive_mobile/main.dart';
import 'package:safedrive_mobile/models.dart';
import 'package:safedrive_mobile/screens/about_screen.dart';
import 'package:safedrive_mobile/state/app_state.dart';
import 'package:safedrive_mobile/widgets/common.dart';

Map<String, dynamic> userJson(String role) => {
      'id': 7,
      'username': 'Test $role',
      'email': '$role@example.com',
      'role': role,
      'email_verified': true,
      'needs_terms_acceptance': false,
      'membership': {'cooperative_name': 'Coop A'},
      'badge': {'verified': false, 'checks': []},
      'verification': role == 'driver' || role == 'umusare' ? {'status': 'PENDING'} : null,
    };

http.Response json(Object body, [int status = 200]) =>
    http.Response(jsonEncode(body), status, headers: {'content-type': 'application/json'});

/// A fake backend implementing the endpoints these tests touch (paths as the real Flask API).
MockClient fakeBackend({String role = 'driver', List<http.Request>? log}) => MockClient((request) async {
      log?.add(request);
      final path = request.url.path.replaceFirst(kApiPrefix, '');
      final authed = request.headers['Authorization'] == 'Bearer tok-123';
      switch (path) {
        case '/auth/login':
          final b = jsonDecode(request.body) as Map<String, dynamic>;
          if (b['password'] != 'Passw0rd!') {
            return json({'error': 'Incorrect email or password.', 'code': 'INVALID_CREDENTIALS'}, 401);
          }
          return json({'token': 'tok-123', 'token_type': 'Bearer', 'user': userJson(role)});
        case '/me':
          return authed ? json({'user': userJson(role)}) : json({'error': 'Sign in again.', 'code': 'UNAUTHORIZED'}, 401);
        case '/auth/logout':
          return json({'signed_out': true});
        case '/admin/overview':
          return json({'users': {'driver': 3, 'admin': 1}, 'cooperatives': {'APPROVED': 2}, 'assistance': []});
        case '/assistance/requests/current':
          return json({'request': null});
      }
      return json({'error': 'Not found.'}, 404);
    });

void main() {
  group('config', () {
    test('normaliseBaseUrl accepts http(s) hosts and strips trailing slashes', () {
      expect(normaliseBaseUrl('http://192.168.1.20:5000/'), 'http://192.168.1.20:5000');
      expect(normaliseBaseUrl(' https://safedrive.example '), 'https://safedrive.example');
      expect(normaliseBaseUrl('ftp://x'), isNull);
      expect(normaliseBaseUrl('192.168.1.20:5000'), isNull);
      expect(normaliseBaseUrl('http://host/?q=1'), isNull);
    });

    test('emulator default never uses localhost', () {
      expect(kDefaultApiUrl.contains('localhost'), isFalse);
    });
  });

  group('ApiClient', () {
    test('sends the bearer token and parses JSON', () async {
      final log = <http.Request>[];
      final api = ApiClient(baseUrl: 'http://srv:5000', client: fakeBackend(log: log), token: 'tok-123');
      final body = await api.get('/me');
      expect(body['user']['role'], 'driver');
      expect(log.single.url.toString(), 'http://srv:5000/api/mobile/v1/me');
      expect(log.single.headers['Authorization'], 'Bearer tok-123');
    });

    test('maps server errors to ApiException and reports expired tokens', () async {
      var expired = 0;
      final api = ApiClient(baseUrl: 'http://srv', client: fakeBackend(), token: 'old', onUnauthorized: () => expired++);
      await expectLater(api.get('/me'), throwsA(isA<ApiException>().having((e) => e.status, 'status', 401)));
      expect(expired, 1);
      await expectLater(
          api.get('/nope'), throwsA(isA<ApiException>().having((e) => e.message, 'message', 'Not found.')));
    });

    test('a non-JSON success (wrong server) is an error, not data', () async {
      final api = ApiClient(baseUrl: 'http://srv', client: MockClient((_) async => http.Response('<html>', 200)));
      await expectLater(api.get('/meta'), throwsA(isA<ApiException>()));
    });
  });

  group('AppState', () {
    test('login stores the token; logout clears it', () async {
      final store = MemorySettingsStore();
      final state = AppState(store: store, httpClient: fakeBackend(role: 'umusare'));
      await state.init();
      expect(state.user, isNull);
      await state.login('umusare@example.com', 'Passw0rd!');
      expect(state.user!.isUmusare, isTrue);
      expect(store.token, 'tok-123');
      await state.logout();
      expect(state.user, isNull);
      expect(store.token, isNull);
    });

    test('a stored token restores the session; a rejected one is discarded', () async {
      final ok = AppState(store: MemorySettingsStore(token: 'tok-123'), httpClient: fakeBackend(role: 'manager'));
      await ok.init();
      expect(ok.user!.isManager, isTrue);
      final store = MemorySettingsStore(token: 'revoked');
      final bad = AppState(store: store, httpClient: fakeBackend());
      await bad.init();
      expect(bad.user, isNull);
      expect(store.token, isNull);
    });

    test('the saved server address is used and validated', () async {
      final store = MemorySettingsStore(apiUrl: 'http://192.168.1.5:5000');
      final state = AppState(store: store, httpClient: fakeBackend());
      await state.init();
      expect(state.apiUrl, 'http://192.168.1.5:5000');
      expect(() => state.setApiUrl('not a url'), throwsArgumentError);
    });
  });

  group('widgets', () {
    testWidgets('assessment labels are shown as produced by the server', (tester) async {
      for (final entry in {
        'SOBER': 'SOBER',
        'UNCERTAIN': 'UNCERTAIN',
        'POTENTIALLY_NOT_SOBER': 'POTENTIALLY NOT SOBER',
      }.entries) {
        await tester.pumpWidget(MaterialApp(
          home: Scaffold(
            body: AssessmentCard(assessment: {
              'assessment': entry.key,
              'valid_frames': 6,
              'total_frames': 7,
              'reasons': entry.key == 'UNCERTAIN' ? ['low_model_confidence'] : <String>[],
            }),
          ),
        ));
        expect(find.text(entry.value), findsOneWidget);
        expect(find.textContaining('model not confident'), entry.key == 'UNCERTAIN' ? findsOneWidget : findsNothing);
      }
    });

    test('no label claims to measure blood alcohol', () {
      for (final label in ['SOBER', 'UNCERTAIN', 'POTENTIALLY_NOT_SOBER', 'ASSESSING', null]) {
        expect(AssessmentStyle.of(label).description.toLowerCase().contains('blood alcohol'), isFalse);
      }
      expect(kAiDisclaimer, contains('not a measurement of blood alcohol'));
    });

    testWidgets('about screen credits the author', (tester) async {
      await tester.pumpWidget(const MaterialApp(home: AboutScreen()));
      expect(find.text('Fabrice NDAYISABA'), findsOneWidget);
      expect(find.text('fabricendayisaba16@gmail.com'), findsOneWidget);
    });

    testWidgets('wrong password shows the server error; right password opens the role dashboard', (tester) async {
      final state = AppState(store: MemorySettingsStore(), httpClient: fakeBackend(role: 'admin'));
      await state.init();
      await tester.pumpWidget(SafeDriveApp(state: state));
      await tester.pumpAndSettle();

      await tester.enterText(find.byKey(const Key('login-email')), 'admin@example.com');
      await tester.enterText(find.byKey(const Key('login-password')), 'wrong');
      await tester.tap(find.byKey(const Key('login-submit')));
      await tester.pumpAndSettle();
      expect(find.text('Incorrect email or password.'), findsOneWidget);

      await tester.enterText(find.byKey(const Key('login-password')), 'Passw0rd!');
      await tester.tap(find.byKey(const Key('login-submit')));
      await tester.pumpAndSettle();
      expect(find.text('Administrator'), findsOneWidget);
      expect(find.textContaining('driver: 3'), findsOneWidget);
    });

    testWidgets('driver dashboard offers monitoring and assistance', (tester) async {
      final state = AppState(store: MemorySettingsStore(token: 'tok-123'), httpClient: fakeBackend());
      await state.init();
      await tester.pumpWidget(SafeDriveApp(state: state));
      await tester.pumpAndSettle();
      expect(find.byKey(const Key('open-monitoring')), findsOneWidget);
      expect(find.byKey(const Key('open-assistance')), findsOneWidget);
      expect(find.text('Request assistance'), findsOneWidget);
    });
  });
}
