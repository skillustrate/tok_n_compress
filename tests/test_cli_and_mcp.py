"""Tests for tok-compress CLI and tok-mcp Model Context Protocol server."""

import os
import json
import tempfile
import pytest
from tok_n_compress.cli import main as cli_main
from tok_n_compress.mcp_server import MCPServer


@pytest.fixture
def temp_db_path():
    temp_dir = tempfile.mkdtemp()
    return os.path.join(temp_dir, "cli_mcp_test.db")


def test_cli_compress_and_query(temp_db_path, capsys):
    """Verify CLI compress, query, and stats commands."""
    messages = [
        {"role": "user", "content": "Critical error code ERR_GATEWAY_TIMEOUT on auth service."},
        {"role": "assistant", "content": "Restarted auth service and scaled pod replicas to 5."},
        {"role": "user", "content": "Is the latency stable now?"},
        {"role": "assistant", "content": "Latency is back under 20ms."}
    ]

    # 1. Compress via CLI
    code = cli_main(["--db", temp_db_path, "compress", "--input", json.dumps(messages), "--keep-recent", "2"])
    assert code == 0
    captured = capsys.readouterr()
    res = json.loads(captured.out)
    assert res["checkpoint_id"] > 0
    assert len(res["updated_history"]) == 3

    # 2. Query via CLI
    code = cli_main(["--db", temp_db_path, "query", "ERR_GATEWAY_TIMEOUT"])
    assert code == 0
    captured = capsys.readouterr()
    matches = json.loads(captured.out)
    assert len(matches) >= 1
    assert "ERR_GATEWAY_TIMEOUT" in matches[0]["content"]

    # 3. Stats via CLI
    code = cli_main(["--db", temp_db_path, "stats"])
    assert code == 0
    captured = capsys.readouterr()
    stats = json.loads(captured.out)
    assert stats["total_checkpoints"] == 1


def test_mcp_server_protocol(temp_db_path):
    """Verify MCP protocol handling: initialize, tools/list, and tools/call."""
    server = MCPServer(db_path=temp_db_path)

    # 1. Initialize
    init_req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    init_resp = server.handle_request(init_req)
    assert init_resp["result"]["serverInfo"]["name"] == "tok_n_compress"

    # 2. Tools list
    list_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    list_resp = server.handle_request(list_req)
    tool_names = [t["name"] for t in list_resp["result"]["tools"]]
    assert "compress_conversation" in tool_names
    assert "query_memory" in tool_names
    assert "rehydrate_segment" in tool_names

    # 3. Call compress tool
    call_req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "compress_conversation",
            "arguments": {
                "messages": [
                    {"role": "user", "content": "Deploying release v2.0.4"},
                    {"role": "assistant", "content": "Release v2.0.4 deployed."}
                ]
            }
        }
    }
    call_resp = server.handle_request(call_req)
    assert "result" in call_resp
    text_content = call_resp["result"]["content"][0]["text"]
    parsed_result = json.loads(text_content)
    assert parsed_result["checkpoint_id"] > 0
