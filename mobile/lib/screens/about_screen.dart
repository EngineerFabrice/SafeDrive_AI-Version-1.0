import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import '../config.dart';
import '../widgets/common.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('About')),
      body: ListView(padding: const EdgeInsets.all(16), children: [
        Icon(Icons.shield, size: 72, color: Theme.of(context).colorScheme.primary),
        const SizedBox(height: 8),
        Text(kAppName, textAlign: TextAlign.center, style: Theme.of(context).textTheme.headlineSmall),
        Text('Mobile app version $kAppVersion', textAlign: TextAlign.center),
        const SizedBox(height: 16),
        SectionCard(title: 'Created by', icon: Icons.person, children: [
          const Text(kAuthor, key: Key('about-author'), style: TextStyle(fontWeight: FontWeight.bold)),
          InkWell(
            onTap: () => launchUrl(Uri(scheme: 'mailto', path: kAuthorEmail)),
            child: const Padding(
              padding: EdgeInsets.symmetric(vertical: 4),
              child: Text(kAuthorEmail, style: TextStyle(decoration: TextDecoration.underline)),
            ),
          ),
          const Text('SafeDrive AI — driver monitoring and cooperative Umusare assistance.'),
        ]),
        const SectionCard(title: 'What the AI does — and does not do', icon: Icons.psychology, children: [
          Text('The camera model looks for visual patterns in the driver\'s face and reports SOBER, UNCERTAIN '
              'or POTENTIALLY NOT SOBER after several frames. It is a research prototype trained on a limited '
              'dataset.'),
          SizedBox(height: 6),
          Text('It does not measure blood alcohol concentration, it is not a medical or legal test, and it '
              'never proves intoxication. If you feel unfit to drive, do not drive: request an Umusare.'),
        ]),
        const SectionCard(title: 'Privacy', icon: Icons.privacy_tip, children: [
          Text('• Camera frames are analysed in memory by the SafeDrive server and are not stored.'),
          Text('• Your location is sent only when you request assistance, go available as an Umusare, or '
              'during an accepted assistance (live sharing with your matched partner only).'),
          Text('• Your sign-in token is kept in the phone\'s secure storage. No server secrets are in the app.'),
        ]),
        const SizedBox(height: 8),
        const Disclaimer(),
      ]),
    );
  }
}
