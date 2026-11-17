"""FastAPI 应用：上传、锁定/解锁、求解作业、确认、下载。

所有会改变「可确认性」的状态变更（核实值、方案发布、确认）都在数据库层
以单个 IMMEDIATE 事务完成；查询接口使用一致快照。后台求解是单线程池，
作业在取输入和发布结果时分别核对修订号与作业状态。
"""
from __future__ import annotations

import csv
import io
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import ValidationError as PydanticValidationError

from . import db
from .parsing import (
    ValidationError as CsvValidationError,
    parse_allowed_csv,
    parse_inspection_csv,
    validate_allowed_dims,
)
from .schemas import AllowedPairsPayload, LockIn
from .solver import LockSpec, solve


# 串行后台求解器：单线程保证求解作业按序执行，作业内部自行做版本检查。
# 每个应用实例（lifespan）持有独立池，避免测试中多实例互相关闭。
_job_pool: Optional[ThreadPoolExecutor] = None


def _pool() -> ThreadPoolExecutor:
    global _job_pool
    if _job_pool is None:
        _job_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="solver")
    return _job_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _job_pool
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="solver")
    _job_pool = pool
    db.init_db()  # 建库 + 重启恢复（未完成作业作废）
    try:
        yield
    finally:
        # 优雅停机：等待本实例的求解线程结束，避免其继续使用数据库（测试中
        # 还会切换数据库路径）。已排队作业若在停机期间发布，仍走完整校验。
        pool.shutdown(wait=True)
        if _job_pool is pool:  # 嵌套的应用实例不得清掉别人的池
            _job_pool = None


app = FastAPI(title="育种巡检数据修复", version="1.0.0", lifespan=lifespan)
STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------- 上传新建会话
@app.post("/api/sessions")
async def create_session(
    inspection: UploadFile = File(...),
    allowed: Optional[UploadFile] = File(None),
    allowed_json: Optional[str] = Form(None),
) -> JSONResponse:
    raw = await inspection.read()
    try:
        parsed_rows = parse_inspection_csv(raw)
        if allowed is not None:
            pairs = parse_allowed_csv(await allowed.read())
        elif allowed_json:
            payload = AllowedPairsPayload.model_validate_json(allowed_json)
            pairs = [(p.variety, p.house) for p in payload.pairs]
            validate_allowed_dims(pairs)
        else:
            raise CsvValidationError("必须提供允许组合（CSV 文件或表单 JSON）。")
    except CsvValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PydanticValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    rows = [r.__dict__ for r in parsed_rows]
    sid = db.create_session(rows, pairs)
    created = db.create_job(sid)
    if created is None:  # 提交后会话被删除（当前 API 无删除路径，防御性处理）
        raise HTTPException(status_code=500, detail="无法创建求解作业。")
    job_id, revision = created
    _pool().submit(_run_job, sid, job_id, revision)
    return JSONResponse(
        status_code=201,
        content={"session_id": sid, "revision": revision, "job_id": job_id},
    )


# ---------------------------------------------------------------- 会话详情
@app.get("/api/sessions/{sid}")
def session_detail(sid: int) -> dict:
    data = db.get_session(sid)  # 单次一致快照
    if data is None:
        raise HTTPException(status_code=404, detail="会话不存在。")
    return _serialize_session(data)


def _serialize_session(data: dict) -> dict:
    sid = data["session"]["id"]
    s, rows, pairs = data["session"], data["rows"], data["pairs"]
    locks = {(l["row_id"], l["field"]): l for l in data["locks"]}
    solution = data["solution"]
    current_rev = s["revision"]

    # 同一快照内的可确认性判断（含修订号、完整方案、核实值快照、确认）。
    issue = db.solution_issue(data)
    download_issue = db.solution_issue(data, require_confirmation=True)
    stale = issue in ("stale", "locks_changed", "solution_mismatch")

    cells = []
    conflicts_out: List[dict] = []
    groups: List[dict] = []
    sol_status: Optional[str] = None
    sol_revision: Optional[int] = None
    total_changes = None
    optimal = False
    objective_proven = False

    if solution is not None:
        sol_status = solution["status"]
        sol_revision = solution["revision"]
        optimal = bool(solution["optimal"])
        objective_proven = bool(solution["objective_proven"])
        total_changes = solution["total_changes"]
        groups = solution["payload"].get("groups", [])
        for c in solution["payload"].get("cells", []):
            cell = dict(c)
            cell["field"] = cell.pop("fld")
            # 仅当方案与当前核实值属于同一修订号时，才用当前锁覆盖展示；
            # 过期方案保持自身生成时的快照，避免把旧方案伪装成新锁的结果。
            lk = locks.get((cell["row_id"], cell["field"]))
            if lk is not None and not stale:
                cell["locked"] = True
                cell["locked_value"] = lk["value"]
                cell["lock_source"] = lk["source"]
                cell["category"] = "locked"
            cells.append(cell)
        conflicts_out = solution["conflicts"]

    job = data["job"]
    return {
        "id": sid,
        "revision": current_rev,
        "row_count": len(rows),
        "variety_count": len({v for v, _ in pairs}),
        "house_count": len({h for _, h in pairs}),
        "created_at": s["created_at"],
        "locked_count": len(locks),
        "job_status": job["status"] if job else None,
        "job_id": job["id"] if job else None,
        "solution_status": sol_status,
        "solution_revision": sol_revision,
        "solution_stale": stale,
        "total_changes": total_changes,
        "optimal": optimal,
        "objective_proven": objective_proven,
        "confirmed": download_issue is None,
        "confirm_block_reason": (
            None if issue is None else db.ISSUE_MESSAGES.get(issue, issue)
        ),
        "download_block_reason": (
            None
            if download_issue is None
            else db.ISSUE_MESSAGES.get(download_issue, download_issue)
        ),
        "rows": [
            {
                "row_id": r["row_id"],
                "line_no": r["line_no"],
                "plant_no": r["plant_no"],
                "tray_no": r["tray_no"],
                "variety_original": r["variety"],
                "house_original": r["house"],
                "height": r["height"],
            }
            for r in rows
        ],
        "allowed_pairs": [{"variety": v, "house": h} for v, h in pairs],
        "locks": data["locks"],
        "cells": cells,
        "conflicts": conflicts_out,
        "groups": groups,
    }


# ---------------------------------------------------------------- 锁定/解锁
@app.post("/api/sessions/{sid}/locks")
def add_lock(sid: int, body: LockIn) -> dict:
    # 校验与写入在同一个事务内完成：校验通过时锁定必然生效，失败时不改任何状态。
    code, revision = db.apply_lock(
        sid, body.row_id, body.field, body.value, body.source
    )
    if code == "no_session":
        raise HTTPException(status_code=404, detail="会话不存在。")
    if code == "row_missing":
        raise HTTPException(status_code=400, detail="row_id 不存在。")
    if code == "original_mismatch":
        row = next(
            r
            for r in db.get_session(sid)["rows"]
            if r["row_id"] == body.row_id
        )
        raise HTTPException(
            status_code=400,
            detail=f"原值锁定的值必须等于原表值「{row[body.field]}」。",
        )
    if code == "out_of_domain":
        raise HTTPException(
            status_code=400,
            detail=f"候选值「{body.value}」不在允许取值域内。",
        )
    return {"revision": revision, "locked": True}


@app.delete("/api/sessions/{sid}/locks/{row_id}/{field}")
def remove_lock(sid: int, row_id: int, field: str) -> dict:
    if field not in ("variety", "house"):
        raise HTTPException(status_code=400, detail="field 必须是 variety 或 house。")
    outcome = db.delete_lock_and_bump(sid, row_id, field)
    if outcome is None:
        raise HTTPException(status_code=404, detail="会话不存在。")
    removed, revision = outcome
    if not removed:
        raise HTTPException(status_code=404, detail="该核实值不存在。")
    return {"revision": revision, "locked": False}


# ---------------------------------------------------------------- 触发求解
@app.post("/api/sessions/{sid}/solve")
def trigger_solve(sid: int) -> dict:
    created = db.create_job(sid)
    if created is None:
        raise HTTPException(status_code=404, detail="会话不存在。")
    job_id, revision = created
    _pool().submit(_run_job, sid, job_id, revision)
    return {"job_id": job_id, "revision": revision}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: int) -> dict:
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="作业不存在。")
    return job


# ---------------------------------------------------------------- 确认方案
@app.post("/api/sessions/{sid}/confirm")
def confirm_solution(sid: int) -> dict:
    data = db.get_session(sid)
    if data is None:
        raise HTTPException(status_code=404, detail="会话不存在。")
    solution = data["solution"]
    if solution is None:
        raise HTTPException(status_code=409, detail=db.ISSUE_MESSAGES["no_solution"])
    # 用快照中的方案 id 作为期望值，事务内若方案已被替换则拒绝，避免确认到
    # 另一个版本/另一份方案的「半次确认」。
    issue = db.confirm_solution(sid, solution["id"])
    if issue is not None:
        status = 404 if issue == "no_session" else 409
        raise HTTPException(
            status_code=status, detail=db.ISSUE_MESSAGES.get(issue, issue)
        )
    confirmed = db.get_session(sid)
    return {
        "confirmed": True,
        "revision": confirmed["session"]["revision"],
        "solution_id": solution["id"],
    }


# ---------------------------------------------------------------- 下载修复 CSV
@app.get("/api/sessions/{sid}/download")
def download_csv(sid: int) -> Response:
    data = db.get_session(sid)  # 原表、方案、确认、核实值来自同一快照
    if data is None:
        raise HTTPException(status_code=404, detail="会话不存在。")
    issue = db.solution_issue(data, require_confirmation=True)
    if issue is not None:
        raise HTTPException(
            status_code=409, detail=db.ISSUE_MESSAGES.get(issue, issue)
        )

    solution = data["solution"]
    cells = {
        (c["row_id"], c["fld"]): c["proposed"]
        for c in solution["payload"]["cells"]
    }
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["行号", "植株号", "托盘号", "品种", "温室", "株高"])
    for r in data["rows"]:
        writer.writerow(
            [
                r["line_no"],
                r["plant_no"],
                r["tray_no"],
                cells[(r["row_id"], "variety")],
                cells[(r["row_id"], "house")],
                r["height"],
            ]
        )
    return Response(
        content="﻿" + buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": (
                f'attachment; filename="repaired_session_{sid}.csv"'
            )
        },
    )


# ---------------------------------------------------------------- 后台作业
def _run_job(sid: int, job_id: int, revision: int) -> None:
    """后台求解。迟到结果（版本已变/作业被取代）直接丢弃，绝不替换新方案。"""
    # 原子认领：仅本会话、本修订、仍为 pending 的作业可以开始。
    if not db.start_job(job_id, sid, revision):
        return
    try:
        # 同一快照读取修订号、原表、允许组合与完整核实值。
        run_input = db.get_run_input(sid, job_id, revision)
        if run_input is None:
            db.finish_job(job_id, "superseded")
            return
        locks = [
            LockSpec(
                row_id=l["row_id"],
                fld=l["field"],
                value=l["value"],
                source=l["source"],
            )
            for l in run_input["locks"]
        ]
        result = solve(run_input["rows"], run_input["pairs"], locks)
        # 发布前在写事务中再次核对修订号与作业状态；失败则整笔回滚。
        published = db.publish_solution(
            sid, job_id, revision, run_input["locks"], result
        )
        if not published:
            db.finish_job(job_id, "superseded")
    except Exception as exc:  # 失败保留原表与核实值，作业标记错误
        db.finish_job(job_id, "error", str(exc))
