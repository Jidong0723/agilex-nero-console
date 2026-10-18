"""Canonical, persisted inference presets and legacy runtime projections."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

SCHEMA = "nero.inference.v2"
PROFILES = ("local5090", "autodl")


def positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def validate_preset(preset):
    host = preset.get("host")
    if not isinstance(host, str) or not host.strip() or any(c.isspace() for c in host):
        raise ValueError("Invalid policy host")
    host = host.strip()
    if "://" in host:
        parsed = urlsplit(host)
        if parsed.scheme not in ("ws", "wss") or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
            raise ValueError("Policy address requires ws:// or wss:// without credentials")
        if parsed.port is not None and parsed.port != preset.get("port"):
            raise ValueError("Address URL port and port field must match")
    port = positive_integer(preset.get("port"), "port")
    if port > 65535:
        raise ValueError("Invalid policy port")
    horizon = positive_integer(preset.get("expected_chunk_steps"), "H")
    prefix = positive_integer(preset.get("replan_steps"), "K")
    if prefix > horizon:
        raise ValueError("K must be no greater than H")
    preset.update(host=host, port=port, expected_chunk_steps=horizon, replan_steps=prefix)


def endpoint_uri(host, port):
    if host.startswith(("ws://", "wss://")):
        parsed = urlsplit(host)
        if parsed.port is not None:
            return host
        hostname = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        return urlunsplit((parsed.scheme, f"{hostname}:{port}", parsed.path, parsed.query, ""))
    return f"ws://{'[' + host + ']' if ':' in host and not host.startswith('[') else host}:{port}"


def canonical(config):
    result = copy.deepcopy(config)
    profiles = result.setdefault("inference_profiles", {})
    active = profiles.setdefault("active", "autodl")
    if active not in PROFILES:
        raise ValueError("Unknown inference profile")
    execution = result.setdefault("execution", {})
    execution.pop('action_delta_scale', None)  # Drop retired settings on migration.
    execution.setdefault("mode", "normal")
    execution.setdefault("rtc_min_replan_steps", 6)
    if execution["mode"] not in ("normal", "rtc"):
        raise ValueError("Unknown inference execution mode")
    positive_integer(execution["rtc_min_replan_steps"], "RTC minimum replan steps")
    if execution["rtc_min_replan_steps"] > 9:
        raise ValueError("RTC minimum replan steps must be <=9 for H20")
    model = result.setdefault("model", {})
    legacy = result.get("schema_version") != SCHEMA
    horizon = execution.get("expected_chunk_steps", 10)
    prefix = execution.get("replan_steps", min(5, horizon))
    for name in PROFILES:
        preset = profiles.setdefault(name, {})
        preset.setdefault("host", "127.0.0.1")
        preset.setdefault("port", 8000 if name == "local5090" else 8001)
        preset.setdefault("checkpoint_label", "")
        preset.setdefault("expected_chunk_steps", horizon)
        preset.setdefault("replan_steps", prefix)
        if legacy and name == active:
            for key in ("host", "port", "checkpoint_label"):
                if key in model:
                    preset[key] = model[key]
            preset.update(expected_chunk_steps=horizon, replan_steps=prefix)
        validate_preset(preset)
    for key in ("host", "port", "checkpoint_label"):
        model.pop(key, None)
    for key in ("expected_chunk_steps", "replan_steps", "max_chunk_steps"):
        execution.pop(key, None)
    result.pop("connection", None)
    result["schema_version"] = SCHEMA
    return result


def project(config):
    result = canonical(config)
    preset = result["inference_profiles"][result["inference_profiles"]["active"]]
    result["model"].update({key: preset[key] for key in ("host", "port", "checkpoint_label")})
    result["execution"].update(expected_chunk_steps=preset["expected_chunk_steps"],
                               replan_steps=preset["replan_steps"], max_chunk_steps=preset["expected_chunk_steps"])
    result["connection"] = {"policy_host": preset["host"], "policy_port": preset["port"],
                            "ssh_forward_host": "127.0.0.1", "ssh_forward_port": preset["port"]}
    return result


def prepare_update(config, request):
    if not isinstance(request, dict):
        raise ValueError("Configuration must be an object")
    result = canonical(config)
    profiles = result["inference_profiles"]
    name = request.get("inference_profile", profiles["active"])
    if name not in PROFILES:
        raise ValueError("Unknown inference profile")
    preset = profiles[name]
    endpoint = request.get("policy_endpoint")
    if endpoint is not None:
        if not isinstance(endpoint, dict) or set(endpoint) - {"host", "port"}:
            raise ValueError("Endpoint must contain host and/or port")
        preset.update(endpoint)
    execution = request.get("execution", {})
    if not isinstance(execution, dict):
        raise ValueError("Execution must be an object")
    length_keys = {"expected_chunk_steps", "replan_steps"}
    if length_keys.intersection(execution) and not length_keys.issubset(execution):
        raise ValueError("Update H and K together")
    if "max_chunk_steps" in execution:
        raise ValueError("max_chunk_steps is derived from H")
    preset.update({key: execution[key] for key in length_keys if key in execution})
    validate_preset(preset)
    profiles["active"] = name
    for key, upper in (("arrival_wait_s", 5),):
        if key in execution:
            value = execution[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= upper:
                raise ValueError(f"{key} must be within [0, {upper}]")
            result["execution"][key] = float(value)
    if "mode" in execution:
        if execution["mode"] not in ("normal", "rtc"):
            raise ValueError("Unknown inference execution mode")
        result["execution"]["mode"] = execution["mode"]
    if "rtc_min_replan_steps" in execution:
        steps = positive_integer(execution["rtc_min_replan_steps"], "RTC minimum replan steps")
        if steps > 9:
            raise ValueError("RTC minimum replan steps must be <=9 for H20")
        result["execution"]["rtc_min_replan_steps"] = steps
    if result["execution"]["mode"] == "rtc" and preset["expected_chunk_steps"] != 20:
        raise ValueError("RTC requires H20; apply H=20 together with RTC mode")
    model = request.get("model", {})
    if not isinstance(model, dict):
        raise ValueError("Model must be an object")
    if "prompt" in model:
        prompt = model["prompt"]
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 500:
            raise ValueError("prompt must contain 1-500 characters")
        result["model"]["prompt"] = prompt.strip()
    return project(result)


def save_config(path: Path, config):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(canonical(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
