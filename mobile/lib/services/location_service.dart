import 'package:geolocator/geolocator.dart';

class LocationException implements Exception {
  LocationException(this.message);

  final String message;

  @override
  String toString() => message;
}

class Fix {
  const Fix(this.lat, this.lon, this.accuracy);

  final double lat;
  final double lon;
  final double accuracy;

  Map<String, dynamic> toJson() => {'lat': lat, 'lon': lon, 'accuracy': accuracy};
}

/// One-shot location reads. The app asks for location permission only when the user starts an action
/// that needs it (an assistance request, going available, live sharing during an accepted assistance),
/// and only while the app is in use; there is no background tracking.
class LocationService {
  static Future<Fix> current() async {
    if (!await Geolocator.isLocationServiceEnabled()) {
      throw LocationException('Location is turned off. Turn on location services to continue.');
    }
    var permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
    }
    if (permission == LocationPermission.denied) {
      throw LocationException('Location permission was denied. It is needed to find help near you.');
    }
    if (permission == LocationPermission.deniedForever) {
      throw LocationException('Location permission is blocked. Allow it for SafeDrive AI in the phone settings.');
    }
    final p = await Geolocator.getCurrentPosition(
      locationSettings: const LocationSettings(accuracy: LocationAccuracy.high, timeLimit: Duration(seconds: 25)),
    );
    return Fix(p.latitude, p.longitude, p.accuracy);
  }

  static Future<void> openSettings() => Geolocator.openAppSettings();
}
