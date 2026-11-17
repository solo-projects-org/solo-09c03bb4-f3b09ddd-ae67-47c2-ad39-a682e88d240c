"""交接一致性测试：并发求解、迟到作业、确认/下载门控、原子性、重启一致性。

这些测试刻意制造关键交错，验证：
1. 锁定/解锁生效后旧方案立即失去确认与下载资格，且失败不留半次锁定；
2. 作业只读取同一修订号的完整核实值；
3. 迟到作业不得覆盖较新方案；
4. 确认/下载必须对应同一核实值与完整方案；
5. 重启后结果一致。
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db  # noqa: E402
from app.main import app  # noqa: E402
from app.solver import LockSpec, SolveResult, solve  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

INSPECTION_CSV = (
    "行号,植株号,托盘号,品种,温室,株高\n"
    "1,P1,T1,V1,H1,10.5\n"
    "2,P1,T2,V2,H1,11.0\n"
    "3,P2,T3,V2,H2,9.8\n"
)
ALLOWED_CSV = "品种,温室\nV1,H1\nV1,H2\nV2,H2\n"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("BREEDER_DB_PATH", str(db_file))
    monkeypatch.setattr(db, "DB_PATH", db_file)
    db.init_db()
    with TestClient(app) as c:
        yield c


def upload(client):
    r = client.post(
        "/api/sessions",
        files={
            "inspection": ("insp.csv", INSPECTION_CSV.encode(), "text/csv"),
            "allowed": ("allow.csv", ALLOWED_CSV.encode(), "text/csv"),
        },
    )
    assert r.status_code == 201, r.text
    return r.json()


def wait_job(client, job_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()["status"]
        if status in ("done", "error", "superseded"):
            return status
        time.sleep(0.02)
    raise AssertionError("作业超时未完成")


def wait_ready(client, sid, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = client.get(f"/api/sessions/{sid}").json()
        if d["solution_status"] is not None and d["job_status"] not in (
            "pending",
            "running",
        ):
            return d
        time.sleep(0.02)
    raise AssertionError("方案超时未生成")


# ------------------------------------------------------------------ 锁定交接
def test_unlock_immediately_revokes_confirmation_and_download(client):
    sid = upload(client)["session_id"]
    wait_ready(client, sid)
    assert client.post(f"/api/sessions/{sid}/confirm").status_code == 200
    assert client.get(f"/api/sessions/{sid}/download").status_code == 200

    r = client.post(
        f"/api/sessions/{sid}/locks",
        json={"row_id": 2, "field": "variety", "value": "V2", "source": "original"},
    )
    assert r.status_code == 200
    d = client.get(f"/api/sessions/{sid}").json()
    assert d["solution_stale"] is True
    assert d["confirmed"] is False
    assert d["download_block_reason"]
    assert client.get(f"/api/sessions/{sid}/download").status_code == 409
    assert client.post(f"/api/sessions/{sid}/confirm").status_code == 409

    # 解锁同样立即失效
    client.post(f"/api/sessions/{sid}/solve")
    wait_ready(client, sid)
    assert client.post(f"/api/sessions/{sid}/confirm").status_code == 200
    r = client.delete(f"/api/sessions/{sid}/locks/2/variety")
    assert r.status_code == 200
    assert client.get(f"/api/sessions/{sid}/download").status_code == 409
    assert client.post(f"/api/sessions/{sid}/confirm").status_code == 409


def test_lock_failure_leaves_no_partial_state(client):
    sid = upload(client)["session_id"]
    wait_ready(client, sid)
    before = client.get(f"/api/sessions/{sid}").json()

    # 非法锁定（候选值不在取值域）必须不改变修订号、核实值、确认状态
    r = client.post(
        f"/api/sessions/{sid}/locks",
        json={"row_id": 1, "field": "house", "value": "H9", "source": "candidate"},
    )
    assert r.status_code == 400
    # 原值不匹配
    r = client.post(
        f"/api/sessions/{sid}/locks",
        json={"row_id": 1, "field": "variety", "value": "V2", "source": "original"},
    )
    assert r.status_code == 400
    # 解除不存在的核实值
    r = client.delete(f"/api/sessions/{sid}/locks/1/variety")
    assert r.status_code == 404
    after = client.get(f"/api/sessions/{sid}").json()
    assert after["revision"] == before["revision"]
    assert after["locks"] == before["locks"]
    assert after["confirmed"] == before["confirmed"]
    # 原表可继续重算
    r = client.post(f"/api/sessions/{sid}/solve")
    assert r.status_code == 200
    assert wait_job(client, r.json()["job_id"]) == "done"


# -------------------------------------------------------------- 迟到作业覆盖
def test_late_job_never_overwrites_newer_solution(client, monkeypatch):
    """旧修订作业在新方案发布后才结束：不得写入，也不得回退当前方案。"""
    sid = upload(client)["session_id"]
    wait_ready(client, sid)
    first = client.get(f"/api/sessions/{sid}").json()
    old_solution_id = None
    # 旧修订作业（手工构造，模拟已排队但迟到）
    old_job, old_rev = db.create_job(sid)

    # 锁定 -> 修订号递增，旧作业应被取代
    client.post(
        f"/api/sessions/{sid}/locks",
        json={"row_id": 3, "field": "variety", "value": "V2", "source": "original"},
    )
    r = client.post(f"/api/sessions/{sid}/solve")
    new_job = r.json()["job_id"]
    assert wait_job(client, new_job) == "done"
    d = client.get(f"/api/sessions/{sid}").json()
    assert d["solution_stale"] is False
    new_solution = db.get_solution(sid)

    # 让「迟到」作业以旧修订尝试发布，必须被拒绝且不改动任何状态
    late = SolveResult(
        status="optimal",
        optimal=True,
        total_changes=0,
        objective_proven=True,
        cells=[],
        conflicts=[],
        groups=[],
    )
    assert db.start_job(old_job, sid, old_rev) is False or db.get_job(old_job)["status"] == "superseded"
    assert db.publish_solution(sid, old_job, old_rev, [], late) is False
    assert db.get_solution(sid)["id"] == new_solution["id"]
    assert db.get_job(old_job)["status"] == "superseded"
    assert client.get(f"/api/sessions/{sid}/download").status_code == 409
    assert first["solution_status"] == "optimal"


def test_job_isolates_locks_at_its_own_revision(client, monkeypatch):
    """作业在运行中遇到确认值时，只用作业开始那一刻同一修订的完整核实值。"""
    sid = upload(client)["session_id"]
    wait_ready(client, sid)
    captured = {}

    real_solve = solve

    def slow_solve(rows, pairs, locks):
        captured["locks"] = sorted((l.row_id, l.fld, l.value) for l in locks)
        time.sleep(0.3)  # 给并发锁定留出交错窗口
        return real_solve(rows, pairs, locks)

    monkeypatch.setattr("app.main.solve", slow_solve)
    r = client.post(f"/api/sessions/{sid}/solve")
    job_id = r.json()["job_id"]
    job_rev = r.json()["revision"]

    # 作业运行中修改核实值：修订号递增，旧作业必须被取代而非发布
    client.post(
        f"/api/sessions/{sid}/locks",
        json={"row_id": 2, "field": "variety", "value": "V2", "source": "original"},
    )
    assert db.get_job(job_id)["status"] == "superseded"
    assert captured.get("locks", []) == []  # 作业读到的是它自己修订的快照
    assert db.current_revision(sid) > job_rev
    # 不会有属于新修订的「假方案」出现
    d = client.get(f"/api/sessions/{sid}").json()
    assert d["solution_stale"] is True
    assert d["confirmed"] is False
    assert client.post(f"/api/sessions/{sid}/confirm").status_code == 409


# ------------------------------------------------------------ 确认/下载门控
def test_confirm_and_download_require_whole_solution(client):
    sid = upload(client)["session_id"]
    wait_ready(client, sid)
    payload = db.get_solution(sid)["payload"]
    assert len(payload["cells"]) == 6
    assert payload["locks"] == []

    # 篡改成不完整方案（缺少一格）：不可确认，也不可下载
    broken = dict(payload)
    broken["cells"] = payload["cells"][:-1]
    with db._write_lock, db.get_conn() as conn:
        conn.execute(
            "UPDATE solutions SET payload=? WHERE session_id=?",
            (__import__("json").dumps(broken), sid),
        )
    d = client.get(f"/api/sessions/{sid}").json()
    assert d["confirm_block_reason"]
    assert client.post(f"/api/sessions/{sid}/confirm").status_code == 409
    assert client.get(f"/api/sessions/{sid}/download").status_code == 409


def test_confirmation_survives_only_matching_revision(client):
    sid = upload(client)["session_id"]
    wait_ready(client, sid)
    assert client.post(f"/api/sessions/{sid}/confirm").status_code == 200
    # 直接改修订号（模拟其他进程改动核实值）：确认立即失效
    with db._write_lock, db.get_conn() as conn:
        conn.execute("UPDATE sessions SET revision = revision + 1 WHERE id=?", (sid,))
    assert client.get(f"/api/sessions/{sid}/download").status_code == 409
    assert client.post(f"/api/sessions/{sid}/confirm").status_code == 409
    # 重算后新方案可确认
    r = client.post(f"/api/sessions/{sid}/solve")
    wait_job(client, r.json()["job_id"])
    assert client.post(f"/api/sessions/{sid}/confirm").status_code == 200
    assert client.get(f"/api/sessions/{sid}/download").status_code == 200


# ---------------------------------------------------------------- 重启一致性
def test_restart_preserves_confirmed_download(client):
    sid = upload(client)["session_id"]
    wait_ready(client, sid)
    assert client.post(f"/api/sessions/{sid}/confirm").status_code == 200
    before = client.get(f"/api/sessions/{sid}/download")
    assert before.status_code == 200

    # 重启：重新执行 init_db 并重建应用实例（同一数据库文件）
    db.init_db()
    with TestClient(app) as c2:
        d = c2.get(f"/api/sessions/{sid}").json()
        assert d["confirmed"] is True
        assert d["solution_stale"] is False
        assert d["revision"] == 1
        after = c2.get(f"/api/sessions/{sid}/download")
        assert after.status_code == 200
        assert after.content == before.content


def test_restart_marks_inflight_jobs_superseded_and_keeps_state(client):
    sid = upload(client)["session_id"]
    wait_ready(client, sid)
    # 手工留下一个「运行中」的作业模拟崩溃/重启
    with db._write_lock, db.get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO jobs(session_id, revision, status, created_at) "
            "VALUES (?,?, 'running', ?)",
            (sid, db.current_revision(sid), db._now()),
        )
        stuck = cur.lastrowid
    db.init_db()
    assert db.get_job(stuck)["status"] == "superseded"
    # 该作业不可能再发布
    late = SolveResult(status="optimal", optimal=True, total_changes=0,
                       objective_proven=True, cells=[], conflicts=[], groups=[])
    assert db.publish_solution(sid, stuck, db.current_revision(sid), [], late) is False
    d = client.get(f"/api/sessions/{sid}").json()
    assert d["solution_status"] == "optimal"
    assert d["solution_stale"] is False


# ------------------------------------------------------------- 并发锁定交错
def test_concurrent_locks_and_solves_keep_invariants(client):
    """多个线程同时锁定/重算：每份方案都能与其修订号的核实值快照对上。"""
    sid = upload(client)["session_id"]
    wait_ready(client, sid)
    errors = []

    def locker(row_id, value):
        try:
            r = client.post(
                f"/api/sessions/{sid}/locks",
                json={
                    "row_id": row_id,
                    "field": "variety",
                    "value": value,
                    "source": "candidate",
                },
            )
            if r.status_code not in (200, 400):
                errors.append(("lock", r.status_code, r.text))
        except Exception as exc:  # pragma: no cover
            errors.append(("lock-exc", repr(exc)))

    def solver():
        try:
            r = client.post(f"/api/sessions/{sid}/solve")
            if r.status_code != 200:
                errors.append(("solve", r.status_code, r.text))
        except Exception as exc:  # pragma: no cover
            errors.append(("solve-exc", repr(exc)))

    threads = [
        threading.Thread(target=locker, args=(1, "V1")),
        threading.Thread(target=locker, args=(2, "V2")),
        threading.Thread(target=solver),
        threading.Thread(target=locker, args=(3, "V2")),
        threading.Thread(target=solver),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []

    deadline = time.time() + 30
    while time.time() < deadline:
        d = client.get(f"/api/sessions/{sid}").json()
        if d["job_status"] not in ("pending", "running"):
            break
        time.sleep(0.02)
    d = client.get(f"/api/sessions/{sid}").json()
    # 方案要么不存在，要么属于当前修订号且与当前核实值一致
    if d["solution_status"] is not None and d["solution_stale"] is False:
        assert d["confirm_block_reason"] is None or d["solution_status"] != "optimal"
        # 当前核实值必须原样保留在方案里
        locks = {(l["row_id"], l["field"]): l["value"] for l in d["locks"]}
        for c in d["cells"]:
            if c["field"] == "variety" and (c["row_id"], "variety") in locks:
                assert c["proposed"] == locks[(c["row_id"], "variety")]
    # 会话依然可用：原表与核实值都在
    assert len(d["rows"]) == 3
    assert len(d["locks"]) == 3


# ---------------------------------------------------------------- 一致性快照
def test_solution_payload_records_locks_snapshot(client):
    sid = upload(client)["session_id"]
    wait_ready(client, sid)
    client.post(
        f"/api/sessions/{sid}/locks",
        json={"row_id": 2, "field": "variety", "value": "V2", "source": "candidate"},
    )
    r = client.post(f"/api/sessions/{sid}/solve")
    wait_job(client, r.json()["job_id"])
    sol = db.get_solution(sid)
    assert {"row_id": 2, "field": "variety", "value": "V2", "source": "candidate"} in sol[
        "payload"
    ]["locks"]
    assert sol["revision"] == db.current_revision(sid)
