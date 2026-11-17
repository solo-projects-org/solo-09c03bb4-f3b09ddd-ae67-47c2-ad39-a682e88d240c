"""SQLite 持久化层。

每次操作打开短连接。写操作统一使用 ``BEGIN IMMEDIATE`` 事务，
使「核实值 + 修订号 + 旧确认作废」与「方案发布 + 旧确认作废」成为不可分割的
一步；读操作使用显式读事务，保证一次响应内的会话、核实值、方案、确认来自
同一快照。这样可以满足：

* 锁定/解锁生效后，旧方案在同一提交中失去确认与下载资格；
* 求解作业只读取「同一修订号的完整核实值」，发布时再次核对修订号与作业状态；
* 确认/下载必须对应同一修订号的完整方案与同一份核实值快照；
* 任一步骤失败都整体回滚，不会留下半次锁定或半次确认。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .solver import SolveResult

DB_PATH = Path(
    os.environ.get("BREEDER_DB_PATH", Path(__file__).resolve().parent.parent / "data" / "app.db")
)
_write_lock = threading.RLock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # busy_timeout 必须先设置，避免切换/确认 WAL 模式时立即报 database is locked。
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    revision INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rows_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    row_id INTEGER NOT NULL,
    line_no TEXT NOT NULL,
    plant_no TEXT NOT NULL,
    tray_no TEXT NOT NULL,
    variety TEXT NOT NULL,
    house TEXT NOT NULL,
    height TEXT NOT NULL,
    UNIQUE(session_id, row_id)
);
CREATE TABLE IF NOT EXISTS allowed_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    variety TEXT NOT NULL,
    house TEXT NOT NULL,
    UNIQUE(session_id, variety, house)
);
CREATE TABLE IF NOT EXISTS locks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    row_id INTEGER NOT NULL,
    field TEXT NOT NULL CHECK(field IN ('variety','house')),
    value TEXT NOT NULL,
    source TEXT NOT NULL CHECK(source IN ('original','candidate')),
    created_at TEXT NOT NULL,
    UNIQUE(session_id, row_id, field)
);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    revision INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending','running','done','error','superseded')),
    error TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS solutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    revision INTEGER NOT NULL,
    status TEXT NOT NULL,
    optimal INTEGER NOT NULL,
    objective_proven INTEGER NOT NULL,
    total_changes INTEGER,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(session_id)
);
CREATE TABLE IF NOT EXISTS conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    solution_id INTEGER NOT NULL REFERENCES solutions(id),
    kind TEXT NOT NULL,
    label TEXT NOT NULL,
    description TEXT NOT NULL,
    row_id INTEGER,
    field TEXT
);
CREATE TABLE IF NOT EXISTS confirmations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    solution_id INTEGER NOT NULL REFERENCES solutions(id),
    revision INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(session_id)
);
"""

# 确认/下载被拒绝的原因代码 -> 面向用户的中文说明。
ISSUE_MESSAGES = {
    "no_session": "会话不存在。",
    "no_solution": "尚无方案，无法确认或下载。",
    "solution_mismatch": "方案已被新的求解结果替换，请刷新页面后重试。",
    "stale": "方案基于旧版本（核实值已变化），请重新求解后再确认。",
    "infeasible": "当前核实值下无解，不存在可确认的修复方案。",
    "not_optimal": "未证明最优的方案不可确认，请重新求解。",
    "conflicts": "方案带有冲突记录，不可确认或下载。",
    "incomplete": "方案不完整（缺少逐格修复值），不可确认或下载。",
    "locks_changed": "方案的核实值快照与当前核实值不一致，请重新求解。",
    "not_confirmed": "方案尚未确认，无法下载。",
    "confirmation_stale": "确认已失效，请重新求解并确认后再下载。",
}


def init_db() -> None:
    """建库并做重启恢复：不可能再被调度的作业标记为 superseded。

    已完成的方案、核实值、确认仍按原修订号保留；重启后读到的确认/方案状态
    与重启前一致（仍由同一套校验决定是否可用）。
    """
    with _write_lock, get_conn() as conn:
        conn.executescript(SCHEMA)
        conn.execute(
            "UPDATE jobs SET status='superseded', "
            "error=COALESCE(error, '服务重启，未完成的作业已取消'), "
            "finished_at=COALESCE(finished_at, ?) "
            "WHERE status IN ('pending','running')",
            (_now(),),
        )
        # 指向已被替换方案或旧修订号的确认永远不可能再通过校验，直接清理，
        # 保证重启后的「已确认」状态与在线时一致。
        conn.execute(
            "DELETE FROM confirmations WHERE NOT EXISTS ("
            "  SELECT 1 FROM sessions s, solutions sol"
            "  WHERE s.id=confirmations.session_id"
            "    AND s.revision=confirmations.revision"
            "    AND sol.id=confirmations.solution_id"
            "    AND sol.session_id=confirmations.session_id"
            "    AND sol.revision=confirmations.revision)"
        )


# ---------------------------------------------------------------- 读取快照
def _load_state(conn: sqlite3.Connection, sid: int) -> Optional[dict]:
    """在调用方打开的读/写事务内读取完整会话状态。"""
    s = conn.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    if s is None:
        return None
    rows = conn.execute(
        "SELECT * FROM rows_data WHERE session_id=? ORDER BY row_id", (sid,)
    ).fetchall()
    pairs = conn.execute(
        "SELECT variety, house FROM allowed_pairs WHERE session_id=? ORDER BY id",
        (sid,),
    ).fetchall()
    locks = conn.execute(
        "SELECT row_id, field, value, source FROM locks WHERE session_id=? "
        "ORDER BY row_id, field",
        (sid,),
    ).fetchall()
    job = conn.execute(
        "SELECT id, revision, status, error FROM jobs WHERE session_id=? "
        "ORDER BY id DESC LIMIT 1",
        (sid,),
    ).fetchone()
    sol_row = conn.execute(
        "SELECT * FROM solutions WHERE session_id=?", (sid,)
    ).fetchone()
    solution = None
    if sol_row is not None:
        solution = dict(sol_row)
        solution["payload"] = json.loads(solution["payload"])
        solution["conflicts"] = [
            dict(c)
            for c in conn.execute(
                "SELECT kind, label, description, row_id, field FROM conflicts "
                "WHERE solution_id=?",
                (sol_row["id"],),
            ).fetchall()
        ]
    confirmation = conn.execute(
        "SELECT solution_id, revision FROM confirmations WHERE session_id=?",
        (sid,),
    ).fetchone()
    return {
        "session": dict(s),
        "rows": [dict(r) for r in rows],
        "pairs": [(p["variety"], p["house"]) for p in pairs],
        "locks": [dict(x) for x in locks],
        "job": dict(job) if job else None,
        "solution": solution,
        "confirmation": dict(confirmation) if confirmation else None,
    }


def get_session(sid: int) -> Optional[dict]:
    """一致快照读取：会话、原表、核实值、方案、确认均来自同一时刻。"""
    with get_conn() as conn:
        conn.execute("BEGIN")
        try:
            return _load_state(conn, sid)
        finally:
            conn.rollback()


def get_solution(sid: int) -> Optional[dict]:
    data = get_session(sid)
    return data["solution"] if data else None


def get_confirmation(sid: int) -> Optional[dict]:
    data = get_session(sid)
    return data["confirmation"] if data else None


def current_revision(sid: int) -> Optional[int]:
    with get_conn() as conn:
        conn.execute("BEGIN")
        try:
            row = conn.execute(
                "SELECT revision FROM sessions WHERE id=?", (sid,)
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.rollback()


# ---------------------------------------------------------------- 校验
def _locks_signature_from_rows(locks: Sequence[dict]) -> Tuple[Tuple[int, str, str, str], ...]:
    return tuple(
        sorted(
            (int(x["row_id"]), str(x["field"]), str(x["value"]), str(x["source"]))
            for x in locks
        )
    )


def _locks_signature_from_payload(payload: dict) -> Optional[Tuple[Tuple[int, str, str, str], ...]]:
    """方案生成时使用的核实值快照；旧版本方案没有该字段时返回 None。"""
    snapshot = payload.get("locks")
    if snapshot is None:
        return None
    if not isinstance(snapshot, list):
        return ()
    try:
        return tuple(
            sorted(
                (
                    int(x["row_id"]),
                    str(x["field"]),
                    str(x["value"]),
                    str(x["source"]),
                )
                for x in snapshot
            )
        )
    except (KeyError, TypeError, ValueError):
        return ()


def payload_is_complete(
    payload: dict,
    rows: Sequence[dict],
    pairs,
    locks: Sequence[dict] = (),
) -> bool:
    """逐格方案必须覆盖每一行、满足全部修复规则，并落实每一份核实值。

    校验内容：每行都有品种与温室；取值来自允许组合；每行组合合法；同株品种
    一致；同托盘温室一致；当前核实值在方案中未被改动。下载的是整份方案，
    任何一项不完整都不能确认或导出。
    """
    cells = payload.get("cells")
    if not isinstance(cells, list):
        return False
    row_ids = {int(r["row_id"]) for r in rows}
    varieties = {v for v, _ in pairs}
    houses = {h for _, h in pairs}
    pair_set = set(pairs)
    proposed: dict = {}
    for c in cells:
        if not isinstance(c, dict):
            return False
        try:
            rid = int(c["row_id"])
        except (KeyError, TypeError, ValueError):
            return False
        fld = c.get("fld")
        if rid not in row_ids or fld not in ("variety", "house"):
            return False
        key = (rid, fld)
        if key in proposed:
            return False
        val = c.get("proposed")
        if not isinstance(val, str) or not val:
            return False
        if fld == "variety" and val not in varieties:
            return False
        if fld == "house" and val not in houses:
            return False
        proposed[key] = val
    if len(proposed) != 2 * len(rows):
        return False

    plant_values: dict = {}
    tray_values: dict = {}
    for r in rows:
        rid = int(r["row_id"])
        v = proposed[(rid, "variety")]
        h = proposed[(rid, "house")]
        if (v, h) not in pair_set:
            return False
        if plant_values.setdefault(r["plant_no"], v) != v:
            return False
        if tray_values.setdefault(r["tray_no"], h) != h:
            return False

    for lk in locks:
        if proposed.get((int(lk["row_id"]), str(lk["field"]))) != str(lk["value"]):
            return False
    return True


def solution_issue(
    data: dict,
    *,
    solution_id: Optional[int] = None,
    require_confirmation: bool = False,
) -> Optional[str]:
    """返回当前方案不可确认/下载的原因代码；None 表示可作为依据。

    ``data`` 必须是 ``get_session`` 的一致快照（或写事务内 ``_load_state`` 的结果）。
    """
    solution = data["solution"]
    if solution is None:
        return "no_solution"
    if solution_id is not None and solution["id"] != solution_id:
        return "solution_mismatch"
    if solution["revision"] != data["session"]["revision"]:
        return "stale"
    if solution["status"] == "infeasible":
        return "infeasible"
    if solution["status"] != "optimal" or not solution["optimal"]:
        return "not_optimal"
    if not solution["objective_proven"]:
        return "not_optimal"
    if solution["conflicts"]:
        return "conflicts"
    if not payload_is_complete(
        solution["payload"], data["rows"], data["pairs"], data["locks"]
    ):
        return "incomplete"
    expected = _locks_signature_from_payload(solution["payload"])
    if expected is not None and expected != _locks_signature_from_rows(data["locks"]):
        return "locks_changed"
    if require_confirmation:
        conf = data["confirmation"]
        if conf is None:
            return "not_confirmed"
        if (
            conf["revision"] != data["session"]["revision"]
            or conf["solution_id"] != solution["id"]
        ):
            return "confirmation_stale"
    return None


# ---------------------------------------------------------------- 新建会话
def create_session(rows: Sequence[dict], pairs: Sequence[Tuple[str, str]]) -> int:
    with _write_lock, get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            "INSERT INTO sessions(revision, created_at) VALUES (1, ?)", (_now(),)
        )
        sid = cur.lastrowid
        conn.executemany(
            """INSERT INTO rows_data
               (session_id, row_id, line_no, plant_no, tray_no, variety, house, height)
               VALUES (?,?,?,?,?,?,?,?)""",
            [
                (
                    sid,
                    r["row_id"],
                    r["line_no"],
                    r["plant_no"],
                    r["tray_no"],
                    r["variety"],
                    r["house"],
                    r["height"],
                )
                for r in rows
            ],
        )
        conn.executemany(
            "INSERT INTO allowed_pairs(session_id, variety, house) VALUES (?,?,?)",
            [(sid, v, h) for v, h in pairs],
        )
        return sid


# ---------------------------------------------------------------- 核实值
def _invalidate_all(conn: sqlite3.Connection, sid: int) -> int:
    """修订号 +1，并在同一事务内删除确认、取代未完成作业，返回新修订号。"""
    conn.execute("UPDATE sessions SET revision = revision + 1 WHERE id=?", (sid,))
    conn.execute("DELETE FROM confirmations WHERE session_id=?", (sid,))
    conn.execute(
        "UPDATE jobs SET status='superseded', finished_at=COALESCE(finished_at, ?) "
        "WHERE session_id=? AND status IN ('pending','running')",
        (_now(), sid),
    )
    return conn.execute(
        "SELECT revision FROM sessions WHERE id=?", (sid,)
    ).fetchone()[0]


def apply_lock(
    sid: int, row_id: int, field: str, value: str, source: str
) -> Tuple[str, Optional[int]]:
    """在同一个写事务内校验并写入核实值，使其立即生效。

    校验（行存在、原值匹配、候选值在取值域内）和写入、修订号递增、旧确认
    删除、未完成作业取代处在同一个 IMMEDIATE 事务中：要么全部生效，要么
    全部不生效，不会留下半次锁定。

    返回 ``(code, revision)``：``code == "ok"`` 表示成功；其余为
    ``no_session`` / ``row_missing`` / ``original_mismatch`` / ``out_of_domain``。
    """
    with _write_lock, get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT variety, house FROM rows_data WHERE session_id=? AND row_id=?",
            (sid, row_id),
        ).fetchone()
        if row is None:
            exists = conn.execute(
                "SELECT 1 FROM sessions WHERE id=?", (sid,)
            ).fetchone()
            return ("row_missing" if exists else "no_session", None)
        if source == "original":
            if value != row[field]:
                return ("original_mismatch", None)
        else:
            pairs = conn.execute(
                "SELECT variety, house FROM allowed_pairs WHERE session_id=?", (sid,)
            ).fetchall()
            domain = (
                {p["variety"] for p in pairs}
                if field == "variety"
                else {p["house"] for p in pairs}
            )
            if value not in domain:
                return ("out_of_domain", None)
        conn.execute(
            """INSERT INTO locks(session_id, row_id, field, value, source, created_at)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(session_id, row_id, field)
               DO UPDATE SET value=excluded.value, source=excluded.source,
                             created_at=excluded.created_at""",
            (sid, row_id, field, value, source, _now()),
        )
        return ("ok", _invalidate_all(conn, sid))


def delete_lock_and_bump(sid: int, row_id: int, field: str):
    """解除核实值并使其立即生效。

    返回 ``None`` 表示会话不存在，``(False, rev)`` 表示该核实值本就不存在
    （未产生任何修改），``(True, rev)`` 表示已删除且修订号已递增。
    """
    with _write_lock, get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        exists = conn.execute(
            "SELECT 1 FROM sessions WHERE id=?", (sid,)
        ).fetchone()
        if exists is None:
            return None
        cur = conn.execute(
            "DELETE FROM locks WHERE session_id=? AND row_id=? AND field=?",
            (sid, row_id, field),
        )
        if cur.rowcount == 0:
            rev = conn.execute(
                "SELECT revision FROM sessions WHERE id=?", (sid,)
            ).fetchone()[0]
            return (False, rev)
        return (True, _invalidate_all(conn, sid))


# ---------------------------------------------------------------- 求解作业
def create_job(sid: int) -> Optional[Tuple[int, int]]:
    """按会话当前修订号创建作业；返回 ``(job_id, revision)``。

    读修订号、取代旧作业、插入新作业在同一事务内，作业携带的修订号必然是
    创建时刻的修订号。
    """
    with _write_lock, get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT revision FROM sessions WHERE id=?", (sid,)
        ).fetchone()
        if row is None:
            return None
        revision = row[0]
        conn.execute(
            "UPDATE jobs SET status='superseded', finished_at=COALESCE(finished_at, ?) "
            "WHERE session_id=? AND status IN ('pending','running')",
            (_now(), sid),
        )
        cur = conn.execute(
            "INSERT INTO jobs(session_id, revision, status, created_at) "
            "VALUES (?,?, 'pending', ?)",
            (sid, revision, _now()),
        )
        return (cur.lastrowid, revision)


def start_job(job_id: int, sid: int, revision: int) -> bool:
    """原子认领作业：仅当作业仍是本会话该修订的 pending 作业时转为 running。"""
    with _write_lock, get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            "UPDATE jobs SET status='running' "
            "WHERE id=? AND session_id=? AND revision=? AND status='pending'",
            (job_id, sid, revision),
        )
        return cur.rowcount == 1


def get_run_input(sid: int, job_id: int, revision: int) -> Optional[dict]:
    """在同一读快照中读取修订号、原表、允许组合与完整核实值。

    只有作业仍是该修订的 running 作业、且会话修订号未变时才返回输入，
    因此求解器看到的核实值必然属于同一个修订号。
    """
    with get_conn() as conn:
        conn.execute("BEGIN")
        try:
            row = conn.execute(
                "SELECT revision FROM sessions WHERE id=?", (sid,)
            ).fetchone()
            job = conn.execute(
                "SELECT revision, status FROM jobs WHERE id=? AND session_id=?",
                (job_id, sid),
            ).fetchone()
            if (
                row is None
                or row[0] != revision
                or job is None
                or job["revision"] != revision
                or job["status"] != "running"
            ):
                return None
            rows = conn.execute(
                "SELECT * FROM rows_data WHERE session_id=? ORDER BY row_id", (sid,)
            ).fetchall()
            pairs = conn.execute(
                "SELECT variety, house FROM allowed_pairs WHERE session_id=? ORDER BY id",
                (sid,),
            ).fetchall()
            locks = conn.execute(
                "SELECT row_id, field, value, source FROM locks WHERE session_id=? "
                "ORDER BY row_id, field",
                (sid,),
            ).fetchall()
            return {
                "rows": [dict(r) for r in rows],
                "pairs": [(p["variety"], p["house"]) for p in pairs],
                "locks": [dict(x) for x in locks],
            }
        finally:
            conn.rollback()


def finish_job(job_id: int, status: str, error: Optional[str] = None) -> bool:
    """结束未完成作业；绝不改写已 done/superseded 的作业。"""
    with _write_lock, get_conn() as conn:
        cur = conn.execute(
            "UPDATE jobs SET status=?, error=?, finished_at=? "
            "WHERE id=? AND status IN ('pending','running')",
            (status, error, _now(), job_id),
        )
        return cur.rowcount == 1


def get_job(job_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None


def publish_solution(
    sid: int,
    job_id: int,
    revision: int,
    locks_used: Sequence[dict],
    result: SolveResult,
) -> bool:
    """原子发布方案；仅当前修订号、作业状态都符合时才写入。

    发布条件：会话修订号仍为 ``revision``；作业仍是 running（未被更新作业取代）；
    不存在由更新作业发布的方案（旧作业不得覆盖新方案）。写入方案、替换冲突、
    删除旧确认、标记作业完成均在同一 IMMEDIATE 事务中，失败整体回滚。
    """
    payload = {
        "cells": [c.__dict__ for c in result.cells],
        "groups": result.groups,
        "locks": [dict(x) for x in locks_used],
    }
    with _write_lock, get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        s = conn.execute(
            "SELECT revision FROM sessions WHERE id=?", (sid,)
        ).fetchone()
        job = conn.execute(
            "SELECT revision, status FROM jobs WHERE id=? AND session_id=?",
            (job_id, sid),
        ).fetchone()
        if s is None or s[0] != revision:
            return False
        if job is None or job["revision"] != revision or job["status"] != "running":
            return False
        existing = conn.execute(
            "SELECT job_id FROM solutions WHERE session_id=?", (sid,)
        ).fetchone()
        if existing is not None and existing["job_id"] > job_id:
            return False
        cur = conn.execute(
            "UPDATE jobs SET status='done', error=NULL, finished_at=? "
            "WHERE id=? AND status='running'",
            (_now(), job_id),
        )
        if cur.rowcount != 1:
            return False
        # 先删确认，再删旧方案：确认通过外键引用方案，顺序反了会触发外键失败。
        conn.execute("DELETE FROM confirmations WHERE session_id=?", (sid,))
        conn.execute(
            "DELETE FROM conflicts WHERE solution_id IN "
            "(SELECT id FROM solutions WHERE session_id=?)",
            (sid,),
        )
        conn.execute("DELETE FROM solutions WHERE session_id=?", (sid,))
        cur = conn.execute(
            """INSERT INTO solutions
               (session_id, job_id, revision, status, optimal, objective_proven,
                total_changes, payload, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                sid,
                job_id,
                revision,
                result.status,
                int(result.optimal),
                int(result.objective_proven),
                result.total_changes,
                json.dumps(payload, ensure_ascii=False),
                _now(),
            ),
        )
        sol_id = cur.lastrowid
        conn.executemany(
            """INSERT INTO conflicts(solution_id, kind, label, description, row_id, field)
               VALUES (?,?,?,?,?,?)""",
            [
                (
                    sol_id,
                    c.kind,
                    c.label,
                    c.description,
                    c.row_id,
                    getattr(c, "fld", None),
                )
                for c in result.conflicts
            ],
        )
        return True


# ---------------------------------------------------------------- 确认
def confirm_solution(sid: int, solution_id: int) -> Optional[str]:
    """原子确认；返回 None 表示成功，否则返回 ISSUE_MESSAGES 中的原因代码。

    在写事务内重新读取并校验修订号、方案状态、完整性与核实值快照，
    因此与并发的锁定/解锁、方案发布不会产生「半次确认」。
    """
    with _write_lock, get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        data = _load_state(conn, sid)
        if data is None:
            return "no_session"
        issue = solution_issue(data, solution_id=solution_id)
        if issue is not None:
            return issue
        conn.execute("DELETE FROM confirmations WHERE session_id=?", (sid,))
        conn.execute(
            "INSERT INTO confirmations(session_id, solution_id, revision, created_at) "
            "VALUES (?,?,?,?)",
            (sid, solution_id, data["session"]["revision"], _now()),
        )
        return None

