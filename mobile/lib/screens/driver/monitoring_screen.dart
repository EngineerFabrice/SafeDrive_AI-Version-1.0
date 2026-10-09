import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../../api/api_client.dart';
import '../../state/app_state.dart';
import '../../widgets/common.dart';
import 'assistance_screen.dart';

class MonitoringScreen extends StatelessWidget {
  const MonitoringScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 2,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Driver monitoring'),
          bottom: const TabBar(tabs: [
            Tab(icon: Icon(Icons.phone_android), text: 'Phone camera'),
            Tab(icon: Icon(Icons.videocam), text: 'Vehicle camera'),
          ]),
        ),
        body: const TabBarView(children: [PhoneMonitoringTab(), VehicleMonitoringTab()]),
      ),
    );
  }
}

void _openAlert(BuildContext context) {
  Navigator.push(context, MaterialPageRoute(builder: (_) => const AssistanceScreen(aiAlert: true)));
}

class _AlertBanner extends StatelessWidget {
  const _AlertBanner();

  @override
  Widget build(BuildContext context) {
    return Card(
      color: Colors.red[50],
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          const Text('Safety alert: please do not drive.', style: TextStyle(fontWeight: FontWeight.bold)),
          const Text('Request a verified Umusare to drive you safely.'),
          const SizedBox(height: 8),
          FilledButton.icon(
            style: FilledButton.styleFrom(backgroundColor: Colors.red[700]),
            onPressed: () => _openAlert(context),
            icon: const Icon(Icons.sos),
            label: const Text('Request assistance'),
          ),
        ]),
      ),
    );
  }
}

// ======================================================================== phone camera
/// The phone's front camera takes a still frame about once a second and uploads it to the server, which
/// runs the same detection -> face quality -> MobileNetV3 -> temporal decision pipeline as the vehicle camera.
/// Frames are not stored on the phone or the server.
class PhoneMonitoringTab extends StatefulWidget {
  const PhoneMonitoringTab({super.key});

  @override
  State<PhoneMonitoringTab> createState() => _PhoneMonitoringTabState();
}

class _PhoneMonitoringTabState extends State<PhoneMonitoringTab> with WidgetsBindingObserver {
  CameraController? _camera;
  Map<String, dynamic>? _model;
  Map<String, dynamic>? _result;
  String? _error;
  bool _running = false;
  bool _starting = false;
  int _sent = 0;
  late final ApiClient _api;

  @override
  void initState() {
    super.initState();
    _api = AppScope.read(context).api;
    WidgetsBinding.instance.addObserver(this);
    _loadModel();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _stop(silent: true);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Never keep the camera open in the background.
    if (state == AppLifecycleState.paused || state == AppLifecycleState.inactive) _stop(silent: true);
  }

  Future<void> _loadModel() async {
    try {
      final body = await _api.get('/monitoring/model');
      if (mounted) setState(() => _model = body);
    } on ApiException catch (e) {
      if (mounted) setState(() => _error = e.message);
    }
  }

  Future<void> _start() async {
    setState(() {
      _starting = true;
      _error = null;
      _result = null;
      _sent = 0;
    });
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) throw CameraException('NoCamera', 'This device has no camera.');
      final front = cameras.firstWhere((c) => c.lensDirection == CameraLensDirection.front,
          orElse: () => cameras.first);
      // No audio: the microphone is never requested.
      final controller = CameraController(front, ResolutionPreset.medium,
          enableAudio: false, imageFormatGroup: ImageFormatGroup.jpeg);
      await controller.initialize();          // shows the Android camera permission prompt
      await _api.post('/monitoring/phone/start');
      if (!mounted) {
        await controller.dispose();
        return;
      }
      setState(() {
        _camera = controller;
        _running = true;
      });
      unawaited(_loop());
    } on CameraException catch (e) {
      setState(() => _error = e.code.contains('Access') || e.code.contains('Permission')
          ? 'Camera permission was denied. Allow camera access for SafeDrive AI in the phone settings.'
          : 'Camera error: ${e.description ?? e.code}');
    } on ApiException catch (e) {
      setState(() => _error = e.message);
    } finally {
      if (mounted) setState(() => _starting = false);
    }
  }

  Future<void> _loop() async {
    while (_running && mounted) {
      final camera = _camera;
      if (camera == null || !camera.value.isInitialized) break;
      final started = DateTime.now();
      try {
        final shot = await camera.takePicture();
        final bytes = await shot.readAsBytes();
        unawaited(File(shot.path).delete().catchError((_) => File(shot.path)));   // nothing kept on the phone
        if (!_running) break;
        final result = await _api.postBytes('/monitoring/phone/frame', bytes);
        _sent++;
        if (kDebugMode) {
          // `adb logcat -s flutter` while testing on a phone; release builds never log this.
          final q = result['face_quality'] as Map<String, dynamic>?;
          debugPrint('[SafeDrive] frame $_sent: ${bytes.length ~/ 1024} KB ${result['frame_size']} '
              'face=${result['face_detected']} quality=${q?['score']} ok=${q?['ok']} reasons=${q?['reasons']} '
              'counted=${result['counted']} assessment=${(result['assessment'] as Map?)?['assessment']} '
              '${result['processing_ms']} ms');
        }
        if (mounted) {
          setState(() {
            _result = result;
            _error = null;
          });
        }
      } on ApiException catch (e) {
        if (e.code == 'NOT_STARTED') {
          await _api.post('/monitoring/phone/start').catchError((_) => <String, dynamic>{});
        } else if (e.code != 'TOO_FAST' && mounted) {
          setState(() => _error = e.message);
          if (e.status == 0 || e.status >= 500) await Future<void>.delayed(const Duration(seconds: 2));
        }
      } on CameraException catch (e) {
        if (mounted) setState(() => _error = 'Camera error: ${e.description ?? e.code}');
        await Future<void>.delayed(const Duration(seconds: 1));
      }
      final elapsed = DateTime.now().difference(started);
      const period = Duration(milliseconds: 900);
      if (elapsed < period) await Future<void>.delayed(period - elapsed);
    }
  }

  Future<void> _stop({bool silent = false}) async {
    final wasRunning = _running;
    _running = false;
    final camera = _camera;
    _camera = null;
    if (wasRunning) {
      try {
        await _api.post('/monitoring/phone/stop');
      } on ApiException {
        // the server also closes idle sessions after two minutes
      }
    }
    await camera?.dispose();
    if (!silent && mounted) setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final assessment = _result?['assessment'] as Map<String, dynamic>?;
    final impairment = _result?['impairment'] as Map<String, dynamic>?;
    final quality = _result?['face_quality'] as Map<String, dynamic>?;
    final modelInfo = _model?['model'] as Map<String, dynamic>?;
    final alert = assessment?['assessment'] == 'POTENTIALLY_NOT_SOBER';
    return ListView(padding: const EdgeInsets.all(16), children: [
      if (_model != null && _model!['enabled'] != true)
        const Card(
          child: ListTile(
            leading: Icon(Icons.warning_amber),
            title: Text('No impairment model is configured on the server'),
            subtitle: Text('Detection runs, but no sobriety assessment can be made (MODEL_PROVIDER=none).'),
          ),
        ),
      if (_model != null && _model!['enabled'] == true && _model!['available'] != true)
        Card(
          child: ListTile(
            leading: const Icon(Icons.error_outline),
            title: const Text('The impairment model is unavailable on the server'),
            subtitle: Text('${_model!['unavailable_reason'] ?? ''}'),
          ),
        ),
      if (_camera != null && _camera!.value.isInitialized)
        ClipRRect(
          borderRadius: BorderRadius.circular(12),
          child: AspectRatio(aspectRatio: 3 / 4, child: CameraPreview(_camera!)),
        )
      else
        Container(
          height: 220,
          alignment: Alignment.center,
          decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.surfaceContainerHighest, borderRadius: BorderRadius.circular(12)),
          child: const Text('Camera off.\nPlace the phone facing you, head and shoulders in view.',
              textAlign: TextAlign.center),
        ),
      const SizedBox(height: 12),
      if (!_running)
        FilledButton.icon(
          key: const Key('phone-start'),
          onPressed: _starting ? null : _start,
          icon: const Icon(Icons.play_arrow),
          label: Text(_starting ? 'Starting…' : 'Start phone monitoring'),
        )
      else
        OutlinedButton.icon(
          key: const Key('phone-stop'),
          onPressed: () => _stop(),
          icon: const Icon(Icons.stop),
          label: const Text('Stop'),
        ),
      if (_error != null)
        Padding(
          padding: const EdgeInsets.only(top: 8),
          child: Text(_error!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
        ),
      const SizedBox(height: 12),
      if (_running || _result != null) AssessmentCard(assessment: assessment),
      if (alert) const _AlertBanner(),
      if (_result != null)
        SectionCard(title: 'Latest frame', icon: Icons.analytics, children: [
          InfoRow('Frames analysed', '$_sent'),
          InfoRow('Driver / face', '${_result!['driver_detected'] == true ? 'driver' : 'no driver'} · '
              '${_result!['face_detected'] == true ? 'face' : 'no face'}'),
          if (quality != null)
            InfoRow('Face quality', '${quality['score']} (needs ${_result!['min_quality'] ?? 0.5})'
                '${quality['ok'] == true ? '' : ' · rejected: ${(quality['reasons'] as List?)?.join(', ')}'}'),
          InfoRow('Counted in assessment', _result!['counted'] == true ? 'yes' : 'no'),
          if (impairment != null)
            InfoRow('Frame model output', '${impairment['prediction'] ?? impairment['status']}'),
          InfoRow('Server time', '${_result!['processing_ms']} ms'),
          if ((_result!['message'] as String?)?.isNotEmpty == true) Text('${_result!['message']}'),
        ]),
      if (modelInfo != null)
        Text('Model: ${modelInfo['name']} v${modelInfo['version']}'
            '${_model!['development_only'] == true ? ' (development prototype)' : ''}',
            style: Theme.of(context).textTheme.bodySmall),
      const SizedBox(height: 8),
      Disclaimer(text: (_model?['notice'] as String?) ?? 'A single frame never decides; the result is based on several '
          'recent frames. It is not a blood alcohol measurement.'),
    ]);
  }
}

// ======================================================================== server-attached camera
/// Controls the camera engine attached to the SafeDrive server (the in-vehicle unit used by the web
/// dashboard). Only the driver who started it sees its results.
class VehicleMonitoringTab extends StatefulWidget {
  const VehicleMonitoringTab({super.key});

  @override
  State<VehicleMonitoringTab> createState() => _VehicleMonitoringTabState();
}

class _VehicleMonitoringTabState extends State<VehicleMonitoringTab> {
  Map<String, dynamic>? _status;
  String? _error;
  bool _busy = false;
  Timer? _poll;

  @override
  void initState() {
    super.initState();
    _refresh();
    _poll = Timer.periodic(const Duration(seconds: 2), (_) => _refresh());
  }

  @override
  void dispose() {
    _poll?.cancel();
    super.dispose();
  }

  Future<void> _refresh() async {
    try {
      final body = await AppScope.read(context).api.get('/monitoring/vehicle/status');
      if (mounted) {
        setState(() {
          _status = body;
          _error = null;
        });
      }
    } on ApiException catch (e) {
      if (mounted) setState(() => _error = e.message);
    }
  }

  Future<void> _control(String action) async {
    setState(() => _busy = true);
    try {
      await AppScope.read(context).api.post('/monitoring/vehicle/$action');
      await _refresh();
    } on ApiException catch (e) {
      if (mounted) showMessage(context, e.message, error: true);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final s = _status;
    final running = s?['running'] == true;
    final mine = s?['owned_by_me'] == true;
    final assessment = s?['assessment'] as Map<String, dynamic>?;
    return ListView(padding: const EdgeInsets.all(16), children: [
      const Text('Uses the camera connected to the SafeDrive server / in-vehicle unit (the same engine as the '
          'web monitoring dashboard).'),
      const SizedBox(height: 12),
      if (_error != null) Text(_error!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
      if (s != null) ...[
        SectionCard(
          title: 'Vehicle camera engine',
          icon: Icons.videocam,
          trailing: StatusChip(running ? (mine ? 'RUNNING' : 'IN USE') : 'STOPPED'),
          children: [
            if (running && !mine) const Text('Another user is monitoring with this camera.'),
            if (mine) ...[
              InfoRow('Status', s['status'] as String?),
              InfoRow('Camera', s['camera_status'] as String?),
              InfoRow('Driver / face', '${s['driver_detected'] == true ? 'driver' : 'no driver'} · '
                  '${s['face_detected'] == true ? 'face' : 'no face'}'),
              InfoRow('Speed', '${s['fps'] ?? 0} FPS'),
              if ((s['message'] as String?)?.isNotEmpty == true) Text('${s['message']}'),
            ],
          ],
        ),
        if (mine) AssessmentCard(assessment: assessment),
        if (mine && assessment?['assessment'] == 'POTENTIALLY_NOT_SOBER') const _AlertBanner(),
        const SizedBox(height: 12),
        if (!running)
          FilledButton.icon(
            onPressed: _busy ? null : () => _control('start'),
            icon: const Icon(Icons.play_arrow),
            label: const Text('Start vehicle camera'),
          ),
        if (mine)
          OutlinedButton.icon(
            onPressed: _busy ? null : () => _control('stop'),
            icon: const Icon(Icons.stop),
            label: const Text('Stop vehicle camera'),
          ),
      ] else if (_error == null)
        const Center(child: CircularProgressIndicator()),
      const SizedBox(height: 12),
      const Disclaimer(),
    ]);
  }
}
