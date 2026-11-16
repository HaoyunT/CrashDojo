"""Exercise frozen request dispatch without datasets or a model."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'request_loader_compat_test', ROOT/'experiments/support/request_loader_compat.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class RequestLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root/'compatibility_sources.json'
        self.output = self.root/'audit'
        self.record = dict(scene_id='scene', adversary_vehicle_id='actor', start_step_10hz=5)
        self.native_calls = []

        def native(path, record):
            request = json.loads(Path(path).read_text())
            self.native_calls.append(request)
            if request['geometry_status'] != 'native_map_verified':
                raise ValueError('Native validation failed')
            return request
        self.module = SimpleNamespace(load_request=native, json=json, Path=Path)

    def request(self, fallback=False):
        request = dict(self.record, family='C1', geometry_status='native_map_verified')
        if fallback:
            request.update(selection_mode='random_fallback', geometry_status='fallback_geometry')
        path = self.root/'request.json'
        path.write_text(json.dumps(request))
        return path

    def compatible_source(self, change_actor=False):
        source = self.root/'original.py'
        source.write_text(
            "def load_request(path, record):\n"
            "    request = json.loads(Path(path).read_text())\n" +
            ("    request['adversary_vehicle_id'] = 'different'\n" if change_actor else '') +
            "    request['compatible_loader_used'] = True\n"
            "    return request\n")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.config.write_text(json.dumps(dict(random_loader_source=source.name,
                                              random_loader_sha256=digest)))
        return source, digest

    def test_native_run_needs_no_legacy_configuration(self):
        helper.install(self.module, self.output, self.config)
        result = self.module.load_request(self.request(), self.record)
        self.assertEqual(len(self.native_calls), 1)
        self.assertEqual(result['geometry_status'], 'native_map_verified')
        audit = json.loads((self.output/'request_loader_compatibility.json').read_text())
        self.assertEqual(audit['loader'], 'original_v26_native')
        self.assertIsNone(audit['original_random_loader_sha256'])

    def test_invalid_native_geometry_is_not_rerouted(self):
        self.compatible_source()
        path = self.request()
        request = json.loads(path.read_text()); request['geometry_status'] = 'invalid'
        path.write_text(json.dumps(request))
        helper.install(self.module, self.output, self.config)
        with self.assertRaisesRegex(ValueError, 'Native validation'):
            self.module.load_request(path, self.record)
        self.assertFalse(self.output.exists())

    def test_fallback_preserves_geometry_identity_and_records_digest(self):
        _, digest = self.compatible_source()
        helper.install(self.module, self.output, self.config)
        path = self.request(True)
        result = self.module.load_request(path, self.record)
        self.assertTrue(result['compatible_loader_used'])
        self.assertEqual(result['geometry_status'], 'fallback_geometry')
        self.assertEqual(self.native_calls, [])
        audit = json.loads((self.output/'request_loader_compatibility.json').read_text())
        self.assertEqual(audit['original_random_loader_sha256'], digest)
        self.assertEqual(audit['request_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertFalse(audit['seven_criteria_used_as_dispatch_filter'])

    def test_missing_fallback_source_fails_explicitly(self):
        helper.install(self.module, self.output, self.config)
        with self.assertRaisesRegex(FileNotFoundError, 'original compatible loader'):
            self.module.load_request(self.request(True), self.record)
        self.assertEqual(self.native_calls, [])

    def test_changed_source_digest_is_rejected(self):
        source, _ = self.compatible_source()
        source.write_text(source.read_text()+'\n# modified source\n')
        helper.install(self.module, self.output, self.config)
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            self.module.load_request(self.request(True), self.record)
        self.assertFalse(self.output.exists())

    def test_fallback_cannot_change_selected_actor(self):
        self.compatible_source(change_actor=True)
        helper.install(self.module, self.output, self.config)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.module.load_request(self.request(True), self.record)
        self.assertFalse(self.output.exists())

    def test_request_must_match_the_frozen_manifest(self):
        self.compatible_source()
        helper.install(self.module, self.output, self.config)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.module.load_request(self.request(True), dict(self.record, start_step_10hz=10))

    def test_both_ablation_wrappers_allow_native_only_inputs(self):
        for suite in ('seed1', 'seeds23'):
            with self.subTest(suite=suite):
                spec = importlib.util.spec_from_file_location('wrapper_'+suite,
                    ROOT/'experiments/ablations'/suite/'compat_v6_loader.py')
                wrapper = importlib.util.module_from_spec(spec); spec.loader.exec_module(wrapper)
                module = SimpleNamespace(load_request=lambda path, record: json.loads(Path(path).read_text()))
                wrapper.install(module, self.output/suite)
                self.assertEqual(module.load_request(self.request(), self.record)['adversary_vehicle_id'], 'actor')


if __name__ == '__main__':
    unittest.main()
