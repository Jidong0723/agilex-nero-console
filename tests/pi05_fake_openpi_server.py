"""Temporary local OpenPI-protocol fixture for browser integration testing."""
from __future__ import annotations

import asyncio
import msgpack
import websockets

request_count = 0


async def handler(socket):
    global request_count
    await socket.send(msgpack.packb({"server": "pi05-test-fixture"}))
    async for payload in socket:
        message = msgpack.unpackb(payload)
        required = {"observation/image", "observation/wrist_image", "observation/state", "prompt"}
        if not required.issubset(message):
            await socket.send("invalid observation")
            continue
        # Ten-step Action Chunk.  Alternate its Y direction every five
        # inference requests so a 0.2 s replan test remains visible but bounded.
        direction = 0.2 if (request_count // 5) % 2 == 0 else -0.2
        request_count += 1
        step = [0.0, direction, 0.0, 0.0, 0.0, 0.0, 0.0]
        await socket.send(msgpack.packb({"actions": [step[:] for _ in range(10)]}))


async def main() -> None:
    async with websockets.serve(handler, "127.0.0.1", 8000, compression=None, max_size=None):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
