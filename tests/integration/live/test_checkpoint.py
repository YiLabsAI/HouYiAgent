"""Integration tests for checkpoint creation and restoration.

This test requires the backend server to be running.
Run: python -m houyi_studio.server
"""

import asyncio
import json
import os
import time

import websockets


def _checkpoint_ws_uri() -> str:
    port = os.getenv("HOUYI_PORT") or os.getenv("HOUYI_INTEGRATION_BACKEND_PORT") or "8000"
    return f"ws://127.0.0.1:{port}/ws/session/test_session"


async def test_checkpoint_integration():
    """Test complete checkpoint workflow via WebSocket."""

    uri = _checkpoint_ws_uri()

    async with websockets.connect(uri) as websocket:
        print("✅ Connected to WebSocket server")

        # Step 1: Start execution with plan data
        start_command = {
            "command_type": "start_execution",
            "command_id": f"cmd_{int(time.time())}",
            "session_id": "test_session",
            "plan_id": "test_plan",
            "inputs": {
                "plan": {
                    "plan_id": "test_plan",
                    "version": 1,
                    "nodes": [
                        {
                            "node_id": "node_1",
                            "node_type": "llm",
                            "config": {
                                "prompt": "Say hello in one sentence",
                            },
                            "inputs": {},
                            "position": {"x": 100, "y": 100},
                        }
                    ],
                    "edges": [],
                    "entry_node_id": "node_1",
                    "metadata": {},
                },
                "run_settings": {
                    "use_mock_llm": True,
                },
            },
        }

        await websocket.send(json.dumps(start_command))
        print("✅ Sent start execution command")

        # Collect events
        checkpoint_id = None
        execution_id = None
        execution_completed = False
        checkpoint_received = False
        received_event_types: list[str] = []
        last_execution_status = None

        start_time = time.time()
        deadline_seconds = 30.0

        while time.time() - start_time < deadline_seconds:
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
            except TimeoutError:
                elapsed = time.time() - start_time
                print(f"⏳ Waiting for events... ({elapsed:.1f}s)")
                continue

            event = json.loads(response)
            event_type = event.get("event_type")
            received_event_types.append(str(event_type))

            if event_type == "ping":
                await websocket.send(json.dumps({"command_type": "pong"}))
                continue

            print(f"📥 {event_type}: {event.get('message', '')[:50]}")

            if event_type == "execution_status":
                execution_id = event.get("execution_id")
                status = event.get("status")
                last_execution_status = status
                print(f"   Execution: {execution_id}, Status: {status}")
                if status == "completed":
                    execution_completed = True
                elif status == "failed":
                    raise AssertionError(
                        "Execution failed before checkpoint creation. "
                        f"execution_id={execution_id}, received_events={received_event_types}"
                    )

            elif event_type == "checkpoint_created":
                checkpoint_id = event.get("checkpoint_id")
                checkpoint_received = True
                print(f"   ✅ Checkpoint created: {checkpoint_id}")

            elif event_type == "node_status":
                print(f"   Node {event.get('node_id')}: {event.get('status')}")

            elif event_type == "streaming_output":
                if event.get("is_final"):
                    print("   ✅ Streaming complete")

            if execution_completed and checkpoint_received:
                break

        if not checkpoint_received:
            raise AssertionError(
                "No checkpoint_created event received. "
                f"execution_id={execution_id}, last_status={last_execution_status}, "
                f"received_events={received_event_types}"
            )

        if not execution_completed:
            raise AssertionError(
                "Execution did not reach completed status. "
                f"execution_id={execution_id}, last_status={last_execution_status}, "
                f"received_events={received_event_types}"
            )

        # Step 3: Test checkpoint restore
        if checkpoint_id:
            print(f"\n🔄 Testing checkpoint restore: {checkpoint_id}")

            restore_command = {
                "command_type": "restore_checkpoint",
                "command_id": f"cmd_{int(time.time())}",
                "session_id": "test_session",
                "execution_id": execution_id,
                "checkpoint_id": checkpoint_id,
                "replay_mode": "deterministic",
            }

            await websocket.send(json.dumps(restore_command))
            print("✅ Sent restore checkpoint command")

            restore_start = time.time()
            restore_deadline_seconds = 10.0
            restore_confirmed = False
            restore_events: list[str] = []

            while time.time() - restore_start < restore_deadline_seconds:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                except TimeoutError:
                    continue

                event = json.loads(response)
                event_type = event.get("event_type")
                restore_events.append(str(event_type))

                if event_type == "ping":
                    await websocket.send(json.dumps({"command_type": "pong"}))
                    continue

                print(f"📥 {event_type}: {event.get('message', '')[:50]}")

                if event_type == "execution_status":
                    message = event.get("message") or ""
                    if "Restored" in message or "restored" in message:
                        restore_confirmed = True
                        break
                    if event.get("status") == "failed":
                        raise AssertionError(
                            "Checkpoint restore failed. "
                            f"checkpoint_id={checkpoint_id}, received_events={restore_events}"
                        )

            if not restore_confirmed:
                raise AssertionError(
                    "Checkpoint restore was not confirmed by execution_status event. "
                    f"checkpoint_id={checkpoint_id}, received_events={restore_events}"
                )

            print("✅ Checkpoint restore test complete")
        else:
            raise AssertionError("No checkpoint was created")

        print("\n✅ Integration test completed successfully")


if __name__ == "__main__":
    uri = _checkpoint_ws_uri()
    print("=" * 60)
    print("Checkpoint Restore Integration Test")
    print("=" * 60)
    print("\nPrerequisites:")
    print("1. Backend server must be running:")
    print("   python -m houyi_studio.server")
    print(f"2. Server should be accessible at {uri}")
    print("\nStarting test...")
    print("=" * 60)

    try:
        asyncio.run(test_checkpoint_integration())
        print("\n" + "=" * 60)
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("=" * 60)
    except Exception:
        print("\n" + "=" * 60)
        print("❌ INTEGRATION TESTS FAILED")
        print("=" * 60)
        exit(1)
