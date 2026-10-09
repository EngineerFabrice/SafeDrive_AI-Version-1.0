import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

import '../config.dart';

/// An error answered by the backend (or a network failure, with [status] 0).
class ApiException implements Exception {
  ApiException(this.status, this.message, {this.code, this.body});

  final int status;
  final String message;
  final String? code;
  final Map<String, dynamic>? body;

  bool get isUnauthorized => status == 401;

  @override
  String toString() => message;
}

/// Thin JSON client for /api/mobile/v1. The bearer token is the only credential sent.
class ApiClient {
  ApiClient({required this.baseUrl, http.Client? client, this.token, this.onUnauthorized})
      : _http = client ?? http.Client();

  final http.Client _http;

  /// Scheme + host [+ port] of the backend, without a trailing slash.
  String baseUrl;
  String? token;

  /// Called when an authenticated request is answered with 401 (expired / revoked token).
  void Function()? onUnauthorized;

  static const Duration timeout = Duration(seconds: 20);

  Uri uri(String path, [Map<String, String>? query]) =>
      Uri.parse('$baseUrl$kApiPrefix$path').replace(queryParameters: query);

  Map<String, String> _headers({bool json = true}) => {
        'Accept': 'application/json',
        if (json) 'Content-Type': 'application/json',
        if (token != null) 'Authorization': 'Bearer $token',
      };

  Future<Map<String, dynamic>> get(String path, {Map<String, String>? query}) =>
      _send(() => _http.get(uri(path, query), headers: _headers(json: false)));

  Future<Map<String, dynamic>> post(String path, [Map<String, dynamic>? body]) =>
      _send(() => _http.post(uri(path), headers: _headers(), body: jsonEncode(body ?? const {})));

  /// Raw binary upload (used for camera frames).
  Future<Map<String, dynamic>> postBytes(String path, List<int> bytes, {String contentType = 'image/jpeg'}) =>
      _send(() => _http.post(uri(path),
          headers: {..._headers(json: false), 'Content-Type': contentType}, body: bytes));

  Future<Map<String, dynamic>> _send(Future<http.Response> Function() request) async {
    final hadToken = token != null;
    http.Response response;
    try {
      response = await request().timeout(timeout);
    } on TimeoutException {
      throw ApiException(0, 'The server did not answer in time. Check the server address and your network.');
    } on SocketException {
      throw ApiException(0, 'Cannot reach the SafeDrive server at $baseUrl. Check Server settings and that the '
          'backend is running.');
    } on http.ClientException catch (e) {
      throw ApiException(0, 'Network error: ${e.message}');
    } on HandshakeException {
      throw ApiException(0, 'Secure connection to the server failed.');
    }
    return decode(response, hadToken: hadToken);
  }

  Map<String, dynamic> decode(http.Response response, {bool hadToken = false}) {
    Map<String, dynamic>? body;
    try {
      final parsed = jsonDecode(utf8.decode(response.bodyBytes));
      if (parsed is Map<String, dynamic>) body = parsed;
    } on FormatException {
      body = null;
    }
    if (response.statusCode >= 200 && response.statusCode < 300) {
      if (body == null) {
        throw ApiException(response.statusCode, 'Unexpected response from the server (is the address correct?).');
      }
      return body;
    }
    if (response.statusCode == 401 && hadToken) onUnauthorized?.call();
    final message = (body?['error'] as String?) ??
        (response.statusCode == 404
            ? 'This server does not provide the SafeDrive mobile API (HTTP 404). Check the server address.'
            : 'Request failed (HTTP ${response.statusCode}).');
    throw ApiException(response.statusCode, message, code: body?['code'] as String?, body: body);
  }

  void close() => _http.close();
}
