"""Exercise the real miner against loopback pools, without a wallet or GPU.

Run: python tests/cp_pool_session_integration.py path/to/cppminer [fault.so]
Handshake timeouts take 30 seconds, share timeouts 60; cases run concurrently,
the whole run takes about 90 seconds. Needs a CPU build, loopback sockets and
Python 3; no wallet, external pool or GPU.

On Linux, an optional second argument names an LD_PRELOAD library that makes
/dev/urandom unavailable, to check that entropy failures exit with status 1:
    gcc -shared -fPIC tests/cp_entropy_fault.c -ldl -o build/libcp_entropy_fault.so
"""

import base64
import concurrent.futures
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import time


class PoolProbe:
    def __init__(self, binary, extra_args=(), env=None):
        self.binary_dir = tempfile.TemporaryDirectory(prefix="cppminer-pool-test-")
        test_binary = pathlib.Path(self.binary_dir.name) / binary.name
        shutil.copy2(binary, test_binary)
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(2)
        self.listener.settimeout(5)
        self.log = tempfile.TemporaryFile(mode="w+")
        self.process = subprocess.Popen([
            str(test_binary), "--backend", "cpu", "--m", "1", "--n", "1",
            "--threads", "2", "--max-nonce", "1",
            "--wallet", "loopback-test",
            "--pool", f"stratum+tcp://127.0.0.1:{self.listener.getsockname()[1]}",
        ] + list(extra_args), stdout=self.log, stderr=self.log, env=env)
        self.connection = None

    def __enter__(self):
        try:
            self.connection, _ = self.listener.accept()
            self.connection.settimeout(5)
            data = b""
            while b"\n" not in data:
                chunk = self.connection.recv(4096)
                if not chunk:
                    raise AssertionError("miner disconnected before authorize")
                data += chunk
            self.auth_id = json.loads(data.split(b"\n", 1)[0])["id"]
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def send(self, message):
        self.connection.sendall((json.dumps(message) + "\n").encode())

    def output(self):
        self.log.seek(0)
        return self.log.read()

    def wait_reconnect(self, timeout, noise=None):
        self.listener.settimeout(0.1)
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            if noise:
                try:
                    self.send(noise)
                except OSError:
                    noise = None
            try:
                retry, _ = self.listener.accept()
                retry.close()
                return
            except socket.timeout:
                if self.process.poll() is not None:
                    raise AssertionError(self.output())
        raise AssertionError("no reconnect within deadline:\n" + self.output())

    def __exit__(self, *_):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.connection:
            self.connection.close()
        self.listener.close()
        self.log.close()
        self.binary_dir.cleanup()


def rejected_authorize(binary):
    with PoolProbe(binary) as pool:
        pool.send({"id": pool.auth_id, "result": False,
                   "error": {"code": -1, "message": "test rejection"}})
        pool.wait_reconnect(5)
        assert "pool rejected authorization" in pool.output()
    return "rejected authorize reconnects"


def spaced_difficulty(binary):
    with PoolProbe(binary) as pool:
        pool.send({"id": pool.auth_id, "result": True})
        pool.send({"method": "mining.set_difficulty", "params": [123]})
        pool.send({"method": "mining.notify", "params": {
            "job_id": "spaced-difficulty", "header": "00" * 76, "cert_version": 3}})
        until = time.monotonic() + 5
        while "[job] notify id=spaced-difficulty " not in pool.output() and time.monotonic() < until:
            time.sleep(0.01)
        output = pool.output()
        assert "[pool] mining.set_difficulty 123" in output, output
        assert "diff=123.0 (no target in notify)" in output, output
    return "spaced difficulty arrays update the target used by subsequent jobs"


def invalid_pearl_jobs(binary):
    with PoolProbe(binary) as pool:
        pool.send({"id": pool.auth_id, "result": True})
        job = {"job_id": "valid-after-invalid", "header": "00" * 76,
               "target": "00" * 32, "cert_version": 3}
        for changes in ({"target": "zz"}, {"target": ""}, {"target": None},
                        {"job_id": ""}, {"job_id": "j" * 128}, {"header": "g0" * 76}):
            pool.send({"method": "mining.notify", "params": dict(job, **changes)})
        # Reader logging acts as a barrier: all preceding invalid messages were processed.
        pool.send({"method": "mining.set_difficulty", "params": [123]})
        until = time.monotonic() + 5
        while "[pool] mining.set_difficulty 123" not in pool.output() and time.monotonic() < until:
            time.sleep(0.01)
        output = pool.output()
        assert "[pool] mining.set_difficulty 123" in output, output
        assert "[plain] mining job=" not in output, output
        pool.send({"method": "mining.notify", "params": job})
        until = time.monotonic() + 5
        while "[plain] mining job=valid-after-invalid" not in pool.output() and time.monotonic() < until:
            time.sleep(0.01)
        assert "[plain] mining job=valid-after-invalid" in pool.output(), pool.output()
    return "invalid Pearl target, ID or header never starts work; subsequent valid job does"


def quantus_job_buffer(binary):
    # Exercise both writes into main's current-job buffer, including queued jobs.
    with PoolProbe(binary, ("--algo", "quantus")) as pool:
        job = {"job_id": "q" * 127, "mining_hash": "00" * 32,
               "target": "00" * 64, "extranonce": "", "difficulty": 1}
        pool.send({"id": pool.auth_id, "result": {
            "id": "loopback-session", "status": "OK", "job": job}})
        for job_id in (job["job_id"], "next-quantus-job"):
            if job_id != job["job_id"]:
                job = dict(job, job_id=job_id)
                pool.send({"method": "job", "params": job})
            until = time.monotonic() + 10
            while time.monotonic() < until:
                output = pool.output()
                assert pool.process.poll() is None, output
                if f"[qpow] mine job={job_id} " in output:
                    break
                time.sleep(0.05)
            else:
                raise AssertionError(pool.output())
    return "Quantus long job id and queued job fit the current-job buffer"


def quantus_submit_then_change(binary):
    with PoolProbe(binary, ("--algo", "quantus")) as pool:
        job = {"job_id": "quantus-submit", "mining_hash": "00" * 32,
               "target": "ff" * 64, "extranonce": "12" * 32, "difficulty": 1}
        pool.send({"id": pool.auth_id, "result": {"id": "qpow-session", "job": job}})
        data = b""
        while b"\n" not in data:
            chunk = pool.connection.recv(4096)
            assert chunk, pool.output()
            data += chunk
        share = json.loads(data.split(b"\n", 1)[0])
        assert share["method"] == "submit", share
        assert share["id"] > pool.auth_id, share
        assert share["params"]["id"] == "qpow-session", share
        assert share["params"]["job_id"] == job["job_id"], share
        assert len(share["params"]["nonce"]) == 128, share
        assert share["params"]["nonce"].startswith(job["extranonce"]), share
        pool.send({"id": share["id"], "result": {"status": "OK"}, "error": None})
        pool.send({"method": "job", "params": dict(job, job_id="quantus-after-share", target="00" * 64)})
        until = time.monotonic() + 5
        while "[qpow] mine job=quantus-after-share " not in pool.output() and time.monotonic() < until:
            assert pool.process.poll() is None, pool.output()
            time.sleep(0.01)
        output = pool.output()
        assert "[pool] submit response:" in output, output
        assert "[qpow] mine job=quantus-after-share " in output, output
    return "real Quantus CPU submits with the correct session/nonce and changes work after its ACK"


def missing_authorize(binary):
    with PoolProbe(binary) as pool:
        pool.wait_reconnect(35, {"id": pool.auth_id + 999, "result": True})
        assert "no authorize response for 30 s" in pool.output()
    return "wrong-id traffic cannot extend authorize deadline"


def quantus_job_changes(binary):
    with PoolProbe(binary, ("--algo", "quantus")) as pool:
        job = {"job_id": "quantus-same-id", "mining_hash": "ab" * 32,
               "target": "00" * 64, "extranonce": "ab" * 32, "difficulty": 1}
        pool.send({"id": pool.auth_id, "result": {
            "id": "loopback-session", "status": "OK", "job": job}})
        for expected in range(1, 5):
            until = time.monotonic() + 10
            marker = "[qpow] mine job=quantus-same-id "
            while pool.output().count(marker) < expected and time.monotonic() < until:
                assert pool.process.poll() is None, pool.output()
                time.sleep(0.01)
            assert pool.output().count(marker) == expected, pool.output()
            if expected == 1:
                pool.send({"method": "job", "params": dict(job,
                    mining_hash=job["mining_hash"].upper(), extranonce=job["extranonce"].upper())})
                time.sleep(0.2)
                assert pool.output().count(marker) == 1, pool.output()
                job = dict(job, mining_hash=job["mining_hash"][:-2] + "01")
            elif expected == 2:
                job = dict(job, target="00" * 63 + "01")
            elif expected == 3:
                job = dict(job, extranonce="ab" * 31 + "01")
            else:
                break
            pool.send({"method": "job", "params": job})
    return "Quantus full hash, target and extranonce changes restart work; hex case does not"


def quantus_early_job(binary):
    with PoolProbe(binary, ("--algo", "quantus")) as pool:
        job = {"job_id": "quantus-early", "mining_hash": "00" * 32,
               "target": "00" * 64, "extranonce": ""}
        pool.send({"id": pool.auth_id + 999, "result": {
            "id": "wrong-session", "status": "OK", "job": dict(job, job_id="wrong-id-job")}})
        for job_id in ("older-early-job", job["job_id"]):
            # Whitespace and nested params.job are valid protocol forms.
            pool.send({"method": "job", "params": {"job": dict(job, job_id=job_id)}})
        time.sleep(0.3)
        assert "[qpow] mine job=" not in pool.output(), pool.output()
        pool.send({"id": pool.auth_id, "result": {
            "job": dict(job, job_id="older-login-job"), "id": "right-session", "status": "OK"}})
        until = time.monotonic() + 5
        while "[qpow] mine job=quantus-early " not in pool.output() and time.monotonic() < until:
            assert pool.process.poll() is None, pool.output()
            time.sleep(0.01)
        output = pool.output()
        assert "session=right-session first_job=quantus-early" in output, output
        assert "[qpow] mine job=quantus-early " in output, output
        assert "[qpow] mine job=older-login-job " not in output, output
    # Login can acknowledge a session before the first job notification.
    with PoolProbe(binary, ("--algo", "quantus")) as pool:
        pool.send({"id": pool.auth_id, "result": {"id": "right-session"}})
        pool.send({"method": "job", "params": job})
        until = time.monotonic() + 5
        while "[qpow] mine job=quantus-early " not in pool.output() and time.monotonic() < until:
            time.sleep(0.01)
        assert "[qpow] mine job=quantus-early " in pool.output(), pool.output()
    return "Quantus matches login ID and mines the latest early or post-login job after ACK"


def quantus_rejected_login(binary):
    job = {"job_id": "must-not-mine", "mining_hash": "00" * 32, "target": "00" * 64}
    for status, error in (("FAIL", None), ("OK", {"code": -1, "message": "rejected"})):
        with PoolProbe(binary, ("--algo", "quantus")) as pool:
            pool.send({"id": pool.auth_id, "result": {"id": "s", "status": status, "job": job},
                       "error": error})
            pool.wait_reconnect(5)
            assert "Quantus login rejected or malformed" in pool.output(), pool.output()
            assert "[qpow] mine job=" not in pool.output(), pool.output()
    return "Quantus rejects failed status and explicit error even with a session and job"


def quantus_missing_login(binary):
    with PoolProbe(binary, ("--algo", "quantus")) as pool:
        pool.wait_reconnect(35, {"id": pool.auth_id + 999, "result": {"id": "wrong-session"}})
        assert "30 s budget" in pool.output(), pool.output()
        assert "[qpow] mine job=" not in pool.output(), pool.output()
    return "Quantus wrong-id traffic cannot extend the 30-second login deadline"


def missing_first_job(binary):
    with PoolProbe(binary) as pool:
        pool.send({"id": pool.auth_id, "result": True})
        pool.wait_reconnect(35, {"method": "mining.notify",
                                "params": {"job_id": "bad", "header": "00" * 76, "target": "zz"}})
        assert "no valid first job for 30 s" in pool.output()
    return "invalid notifications cannot extend first-job deadline"


def job_before_authorize(binary):
    with PoolProbe(binary) as pool:
        pool.send({"method": "mining.notify", "params": {
            "job_id": "early-job", "header": "00" * 76,
            "target": "00" * 32, "cert_version": 3}})
        time.sleep(0.3)
        assert "attempt timing:" not in pool.output(), "mined before authorize ACK"
        pool.send({"id": pool.auth_id, "result": True, "error": None})
        until = time.monotonic() + 5
        while time.monotonic() < until:
            output = pool.output()
            if "stopped after max_nonce=1" in output:
                return "early job is preserved and mined after accepted authorize"
            time.sleep(0.05)
        raise AssertionError(pool.output())


def changed_job_identity(binary, extra_args=()):
    with PoolProbe(binary, extra_args) as pool:
        pool.send({"id": pool.auth_id, "result": True})
        job = {"job_id": "same-id", "header": "00" * 76,
               "target": "00" * 32, "cert_version": 3}
        for expected in range(1, 5):
            pool.send({"method": "mining.notify", "params": job})
            until = time.monotonic() + 5
            while time.monotonic() < until:
                if pool.output().count("stopped after max_nonce=1") == expected:
                    break
                time.sleep(0.05)
            else:
                raise AssertionError(pool.output())
            if expected == 1:
                # Equivalent duplicate must not restart work (also catches sizeof(pointer)).
                pool.send({"method": "mining.notify", "params": job})
                until = time.monotonic() + 2
                while "duplicate notify ignored" not in pool.output() and time.monotonic() < until:
                    time.sleep(0.05)
                assert "duplicate notify ignored" in pool.output(), pool.output()
                job = dict(job, header="00" * 75 + "01")
            elif expected == 2:
                job = dict(job, target="00" * 31 + "01")
            elif expected == 3:
                job = dict(job, cert_version=2)
    return "duplicate jobs are ignored; full header, target and certificate changes restart work"


def active_job_change(binary, extra_args=()):
    with PoolProbe(binary, ("--max-nonce", "0", *extra_args)) as pool:
        pool.send({"id": pool.auth_id, "result": True})
        job = {"job_id": "active-same-id", "header": "00" * 76,
               "target": "00" * 32, "cert_version": 3}
        pool.send({"method": "mining.notify", "params": job})
        until = time.monotonic() + 15
        while "[gen] nonce=0:" not in pool.output() and time.monotonic() < until:
            assert pool.process.poll() is None, pool.output()
            time.sleep(0.005)
        assert "[gen] nonce=0:" in pool.output(), pool.output()
        pool.send({"method": "mining.notify", "params": dict(job, header="00" * 75 + "01")})
        until = time.monotonic() + 15
        while time.monotonic() < until:
            output = pool.output()
            assert pool.process.poll() is None, output
            if output.count("[plain] mining job=active-same-id...") >= 2:
                assert "cancelling stale work" in output, output
                return "changed header with the same ID cancels active mining and starts queued work"
            time.sleep(0.005)
        raise AssertionError(pool.output())


def receive_share(pool):
    pool.send({"id": pool.auth_id, "result": True})
    pool.send({"method": "mining.notify", "params": {
        "job_id": "submit-test", "header": "00" * 76,
        "target": "0000004000000000" + "00" * 24, "cert_version": 3}})
    data = b""
    while b"\n" not in data:
        chunk = pool.connection.recv(65536)
        assert chunk, pool.output()
        data += chunk
    share = json.loads(data.split(b"\n", 1)[0])
    assert share["method"] == "mining.submit", share
    proof = base64.b64decode(share["params"]["plain_proof"])
    assert len(proof) > 1000, pool.output()
    return share["id"]


def accepted_share(binary):
    with PoolProbe(binary) as pool:
        submit_id = receive_share(pool)
        pool.send({"id": submit_id, "result": True, "error": None})
        pool.listener.settimeout(62)
        try:
            retry, _ = pool.listener.accept()
            retry.close()
            raise AssertionError("acknowledged share caused reconnect:\n" + pool.output())
        except socket.timeout:
            output = pool.output()
            assert pool.process.poll() is None, output
            assert "[pool] submit response:" in output, output
            assert "no pool reply to a submit" not in output, output
    return "share ACK stays connected past the 60-second deadline"


def unacknowledged_share(binary):
    with PoolProbe(binary) as pool:
        submit_id = receive_share(pool)
        pool.wait_reconnect(65, {"id": submit_id + 999, "result": True})
        assert "no pool reply to a submit for 60 s" in pool.output()
    return "unrelated reply traffic cannot acknowledge a share or prevent reconnect"


def wait_for(pool, marker, timeout=15):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        output = pool.output()
        assert pool.process.poll() is None, output
        if marker in output:
            return output
        time.sleep(.01)
    raise AssertionError(pool.output())


def job(job_id, target=True):
    result = {'job_id': job_id, 'header': '00' * 76, 'cert_version': 3}
    if target:
        result['target'] = '00' * 32
    return {'method': 'mining.notify', 'params': result}


def early_two_jobs(binary):
    with PoolProbe(binary, ('--max-nonce', '0')) as pool:
        pool.send(job('early-old'))
        pool.send(job('early-new'))
        pool.send({'method': 'mining.set_difficulty', 'params': [123]})
        wait_for(pool, '[pool] mining.set_difficulty 123')
        pool.send({'id': pool.auth_id, 'result': True})
        wait_for(pool, '[gen] nonce=0:')
        output = pool.output()
        assert '[plain] mining job=early-old' not in output, output
        assert '[plain] mining job=early-new' in output, output
        assert 'mining queued job=early-new' not in output, output
    return 'latest of two pre-authorization jobs starts; old job is not mined'


def difficulty_reconnect(binary):
    with PoolProbe(binary) as pool:
        pool.send({'id': pool.auth_id, 'result': True})
        pool.send({'method': 'mining.set_difficulty', 'params': [123]})
        pool.send(job('session-one', False))
        wait_for(pool, 'stopped after max_nonce=1')
        pool.connection.shutdown(socket.SHUT_RDWR)
        pool.connection.close()
        pool.listener.settimeout(10)
        pool.connection, _ = pool.listener.accept()
        pool.connection.settimeout(10)
        data = b''
        while b'\n' not in data:
            data += pool.connection.recv(4096)
        auth_id = json.loads(data.split(b'\n', 1)[0])['id']
        pool.send({'id': auth_id, 'result': True})
        pool.send(job('session-two', False))
        output = wait_for(pool, '[job] notify id=session-two ')
        assert 'id=session-two header=0000000000000000... diff=32.0' in output, output
    return 'difficulty resets to 32 on reconnect'


def difficulty_job_snapshot(binary):
    with PoolProbe(binary) as pool:
        pool.send({'method': 'mining.set_difficulty', 'params': [32]})
        pool.send(job('difficulty-snapshot', False))
        pool.send({'method': 'mining.set_difficulty', 'params': [123]})
        wait_for(pool, '[pool] mining.set_difficulty 123')
        pool.send({'id': pool.auth_id, 'result': True})
        output = wait_for(pool, '[job] notify id=difficulty-snapshot ')
        assert 'id=difficulty-snapshot header=0000000000000000... diff=32.0' in output, output
    return 'targetless job retains difficulty snapshot from receipt'


def targetless_verify(binary):
    with PoolProbe(binary, ('--verify',)) as pool:
        (pathlib.Path(pool.binary_dir.name) / 'pp_header.bin').mkdir()
        pool.send({'id': pool.auth_id, 'result': True})
        # With the stratum target (0xFFFF << 208) / diff, difficulty 1 lets the 1024^2 attempt
        # find a share; much easier targets fail the proof's penalized-target check.
        pool.send({'method': 'mining.set_difficulty', 'params': [1]})
        pool.send(job('verify-no-target', False))
        pool.connection.settimeout(25)
        data = b''
        while b'\n' not in data:
            chunk = pool.connection.recv(65536)
            assert chunk, pool.output()
            data += chunk
        share = json.loads(data.split(b'\n', 1)[0])
        assert share['method'] == 'mining.submit', share
        output = pool.output()
        assert 'verify OK' in output and 'verify failed' not in output, output
        assert '[mode] verify=1' in output, output
        assert not list(pathlib.Path(pool.binary_dir.name).glob('pp_*_proof.b64')), output
    return '--verify verifies a targetless job before submitting proof'


def unicode_job_id(binary):
    with PoolProbe(binary) as pool:
        pool.send({'id': pool.auth_id, 'result': True})
        raw = json.dumps(job('escaped-1')).replace('escaped-1', 'escaped-\\u0031')
        pool.connection.sendall((raw + '\n').encode())
        pool.send({'method': 'mining.set_difficulty', 'params': [123]})
        output = wait_for(pool, '[pool] mining.set_difficulty 123')
        wait_for(pool, '[plain] mining job=escaped-1')
    return 'valid Unicode escape in job ID is decoded and mined'


def header_path_failure(binary):
    with PoolProbe(binary) as pool:
        (pathlib.Path(pool.binary_dir.name) / 'pp_header.bin').mkdir()
        pool.send({'id': pool.auth_id, 'result': True})
        pool.send(job('header-path-failure'))
        output = wait_for(pool, 'job stopped (max_nonce)')
        assert '[gen] nonce=0:' in output and 'header tmp' not in output, output
    return 'normal mining does not write diagnostic header files'


def invalid_authorize_types(binary):
    for result in (0, "", {"status": "FAIL"}):
        with PoolProbe(binary) as pool:
            pool.send({"id": pool.auth_id, "result": result, "error": None})
            pool.wait_reconnect(5)
            assert "pool rejected authorization" in pool.output(), pool.output()
            assert "[plain] mining job=" not in pool.output(), pool.output()
    return "numeric, string and failed-status authorization results are rejected"


def entropy_failure(binary, library):
    env = dict(os.environ, LD_PRELOAD=str(library))
    for extra in ((), ("--cpu-gen",), ("--algo", "quantus")):
        with PoolProbe(binary, extra, env=env) as pool:
            if "quantus" in extra:
                pool.send({"id": pool.auth_id, "result": {"id": "fault-session", "job": {
                    "job_id": "entropy-failure", "mining_hash": "00" * 32,
                    "target": "00" * 64, "extranonce": "", "difficulty": 1}}})
            else:
                pool.send({"id": pool.auth_id, "result": True})
                pool.send(job("entropy-failure"))
            assert pool.process.wait(timeout=10) == 1, pool.output()
            output = pool.output()
            assert "failure; exiting with status 1" in output, output
            assert "max_nonce" not in output.split("[plain] mining job=")[-1], output
    return "CPU preparation, host preparation and Quantus entropy failures exit with status 1"


def diagnostic_files(binary):
    with tempfile.TemporaryDirectory(prefix="cppminer-diagnostic-test-") as folder:
        test_binary = pathlib.Path(folder) / binary.name
        shutil.copy2(binary, test_binary)
        args = [str(test_binary), "--backend", "cpu", "--mock", "--dry-run", "--verify",
                "--mock-diff", "32", "--m", "1", "--n", "1", "--threads", "1",
                "--max-nonce", "4"]
        processes = [subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                     for _ in range(2)]
        try:
            for process in processes:
                output = process.communicate(timeout=30)[0].decode(errors="replace")
                assert process.returncode == 0 and "verify OK" in output, output
                assert "dry-run: proof saved" in output, output
                prefix = pathlib.Path(folder) / f"pp_{process.pid}_1"
                assert pathlib.Path(str(prefix) + "_header.bin").stat().st_size == 76
                assert pathlib.Path(str(prefix) + "_proof.b64").stat().st_size > 1000
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
        assert len(list(pathlib.Path(folder).glob("pp_*_proof.b64"))) == 2
    return "concurrent dry-run processes save separate verified header/proof pairs"


if __name__ == "__main__":
    binary = pathlib.Path(sys.argv[1]).resolve()
    # Each miner has its own executable directory and proof files.
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(case, binary) for case in (
            early_two_jobs, difficulty_reconnect, difficulty_job_snapshot, targetless_verify,
            unicode_job_id, header_path_failure, invalid_authorize_types, diagnostic_files,
            spaced_difficulty,
            invalid_pearl_jobs,
            quantus_submit_then_change,
            quantus_early_job, quantus_rejected_login, quantus_missing_login,
            quantus_job_changes, active_job_change, changed_job_identity, quantus_job_buffer,
            rejected_authorize, missing_authorize, missing_first_job, job_before_authorize,
            accepted_share, unacknowledged_share)]
        for future in futures:
            print("PASS:", future.result(), flush=True)
    if len(sys.argv) > 2:
        print("PASS:", entropy_failure(binary, pathlib.Path(sys.argv[2]).resolve()), flush=True)
