import 'package:flutter/material.dart';

import '../../api/api_client.dart';
import '../../config.dart';
import '../../state/app_state.dart';
import '../../widgets/common.dart';

/// Where the backend runs. The app never stores anything but this address and its own sign-in token.
class ServerSettingsScreen extends StatefulWidget {
  const ServerSettingsScreen({super.key});

  @override
  State<ServerSettingsScreen> createState() => _ServerSettingsScreenState();
}

class _ServerSettingsScreenState extends State<ServerSettingsScreen> {
  late final TextEditingController _url;
  String? _result;
  bool _ok = false;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _url = TextEditingController(text: AppScope.read(context).apiUrl);
  }

  @override
  void dispose() {
    _url.dispose();
    super.dispose();
  }

  Future<void> _test() async {
    final url = normaliseBaseUrl(_url.text);
    if (url == null) {
      setState(() {
        _ok = false;
        _result = 'Enter an address like http://192.168.1.20:5000';
      });
      return;
    }
    setState(() => _busy = true);
    final client = ApiClient(baseUrl: url);
    try {
      final meta = await client.get('/meta');
      setState(() {
        _ok = true;
        _result = 'Connected: ${meta['app']} API v${meta['api_version']}';
      });
    } on ApiException catch (e) {
      setState(() {
        _ok = false;
        _result = e.message;
      });
    } finally {
      client.close();
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _save() async {
    try {
      await AppScope.read(context).setApiUrl(_url.text);
      if (mounted) {
        showMessage(context, 'Server address saved.');
        Navigator.pop(context);
      }
    } on ArgumentError catch (e) {
      if (mounted) showMessage(context, '${e.message}', error: true);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Server settings')),
      body: ListView(padding: const EdgeInsets.all(16), children: [
        TextField(
          key: const Key('server-url'),
          controller: _url,
          keyboardType: TextInputType.url,
          decoration: const InputDecoration(labelText: 'SafeDrive server address', hintText: 'http://10.0.2.2:5000'),
        ),
        const SizedBox(height: 12),
        Row(children: [
          OutlinedButton.icon(
            onPressed: _busy ? null : _test,
            icon: const Icon(Icons.network_check),
            label: const Text('Test connection'),
          ),
          const SizedBox(width: 12),
          FilledButton.icon(onPressed: _save, icon: const Icon(Icons.save), label: const Text('Save')),
        ]),
        if (_result != null) ...[
          const SizedBox(height: 12),
          Text(_result!, style: TextStyle(color: _ok ? Colors.green[700] : Theme.of(context).colorScheme.error)),
        ],
        const SizedBox(height: 24),
        const SectionCard(title: 'Which address?', icon: Icons.help_outline, children: [
          Text('• Android emulator: http://10.0.2.2:5000 (the computer running the backend).'),
          SizedBox(height: 4),
          Text('• Physical phone: the computer\'s Wi-Fi IP, e.g. http://192.168.1.20:5000. Start the backend '
              'with SAFEDRIVE_HOST=0.0.0.0 and allow port 5000 in the firewall. Phone and computer must be '
              'on the same network.'),
          SizedBox(height: 4),
          Text('• "localhost" on a phone means the phone itself, so it never reaches your computer.'),
          SizedBox(height: 4),
          Text('• Debug builds allow plain HTTP for development; release builds require HTTPS.'),
        ]),
        Text('Build default: $kDefaultApiUrl', style: Theme.of(context).textTheme.bodySmall),
      ]),
    );
  }
}
