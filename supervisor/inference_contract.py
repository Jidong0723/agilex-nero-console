"""The shared NERO state7/action7 OpenPI contract, independent of deployment."""
from __future__ import annotations

import numpy as np


def metadata_compatibility(metadata, horizon):
    if not isinstance(metadata, dict):
        return {"status": "incompatible", "message": "Model metadata must be an object"}
    # valid_action_dim describes the transmitted action; action_dim may describe
    # a model's internal padded dimension when valid_action_dim is provided.
    declarations = [("state_dim", 7), ("valid_action_dim", 7),
                    ("action_horizon", horizon), ("horizon", horizon)]
    if "valid_action_dim" not in metadata:
        declarations.append(("action_dim", 7))
    for key, expected in declarations:
        if key in metadata and (isinstance(metadata[key], bool) or metadata[key] != expected):
            return {"status": "incompatible", "message": f"Model {key}={metadata[key]}, expected {expected}"}
    return {"status": "pending", "message": "待响应验证"}


def make_observation(position, rotvec, ratio, external, wrist, prompt):
    state = np.asarray([*position, *rotvec, ratio], dtype=np.float32)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError("NERO observation requires finite state7")
    for image in (external, wrist):
        if not isinstance(image, np.ndarray) or image.shape != (224, 224, 3) or image.dtype != np.uint8:
            raise ValueError("NERO images must be 224x224 RGB uint8")
    return {"observation/image": external, "observation/wrist_image": wrist,
            "observation/state": state, "prompt": prompt}


def action_rows(response, horizon):
    actions = response.get("actions") if isinstance(response, dict) else None
    if actions is None:
        raise RuntimeError("Model response does not contain actions")
    rows = actions.tolist() if hasattr(actions, "tolist") else actions
    if not isinstance(rows, list) or len(rows) != horizon:
        raise RuntimeError(f"Model action chunk must contain exactly {horizon} steps, got {len(rows) if isinstance(rows, list) else 'invalid'}")
    for row in rows:
        if (not isinstance(row, (list, tuple)) or len(row) != 7
                or any(isinstance(v, (bool, str)) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in row)):
            raise RuntimeError("Model action rows must contain seven finite numbers")
    return rows
