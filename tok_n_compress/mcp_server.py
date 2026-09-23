"""
Model Context Protocol (MCP) Server for tok_n_compress v2.0 (AHMS).

Enables seamless zero-config tool integration across all MCP-compatible agent harnesses:
- Claude Code (`claude mcp add tok-compress ...`)
- Cursor / Windsurf / Zed
- Claude Desktop / Cline / Continue / Roo-Code
- Any JSON-RPC 2.0 stdio MCP client
"""

import sys
import json
import logging
from typing import Dict, Any, Optional

from .skill import ConversationCompressorSkill

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="[tok_n_compress_mcp] %(message)s")


class MCPServer:
    """Standard Model Context Protocol (MCP) stdio server implementation."""

    def __init__(self, db_path: str = "./checkpoints.db"):
        self.skill = ConversationCompressorSkill(db_path=db_path)
        self.tools = [
            {
                "name": "compress_conversation",
                "description": (
                    "Compress a list of conversation messages into a persistent SQLite checkpoint snapshot. "
                    "Replaces older dialogue turns with a structured summary while keeping recent turns in active context. "
                    "Respects pinned turns and snaps to clean turn boundaries."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "description": "List of message objects (with role and content) to compress.",
                            "items": {"type": "object"}
                        },
                        "keep_recent_n": {
                            "type": "integer",
                            "description": "Number of most recent messages to preserve uncompressed (default: 2).",
                            "default": 2
                        },
                        "user_query": {
                            "type": "string",
                            "description": "Optional user query providing context for compression."
                        }
                    },
                    "required": ["messages"]
                }
            },
            {
                "name": "query_memory",
                "description": (
                    "Search compressed long-term memory for relevant past conversation segments, "
                    "error codes, decisions, or files mentioned earlier using Hybrid RRF (FTS5 + Vector Cosine)."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search keyword, technical term, or casual question to find in past conversations."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of relevant segments to retrieve (default: 3).",
                            "default": 3
                        },
                        "use_hybrid": {
                            "type": "boolean",
                            "description": "Whether to use Hybrid RRF search (default: true).",
                            "default": True
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "rehydrate_segment",
                "description": (
                    "Rehydrate a raw conversation segment from long-term memory into active context. "
                    "Wraps historical dialogue in structured XML envelopes (<retrieved_context>) for clean injection."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "segment_id": {
                            "type": "integer",
                            "description": "The segment ID or checkpoint ID retrieved via query_memory."
                        },
                        "user_query": {
                            "type": "string",
                            "description": "The user's current question to format the rehydration context around."
                        },
                        "format_style": {
                            "type": "string",
                            "enum": ["xml", "legacy"],
                            "description": "Formatting style for prompt injection ('xml' or 'legacy', default: 'xml').",
                            "default": "xml"
                        }
                    },
                    "required": ["segment_id"]
                }
            },
            {
                "name": "get_checkpoint_map",
                "description": (
                    "Retrieve the high-level Layer 2 episodic summary map of the conversation. "
                    "Returns a lightweight overview of all checkpoints, timestamps, and files modified."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of checkpoints to include in the map (default: 50).",
                            "default": 50
                        }
                    }
                }
            },
            {
                "name": "pin_segment",
                "description": "Pin or unpin a historical segment to protect it from automated eviction.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "segment_id": {
                            "type": "integer",
                            "description": "The ID of the segment to pin or unpin."
                        },
                        "pinned": {
                            "type": "boolean",
                            "description": "True to pin (protect), False to unpin.",
                            "default": True
                        }
                    },
                    "required": ["segment_id"]
                }
            },
            {
                "name": "get_memory_stats",
                "description": "Get storage statistics, model context window, and dynamic budgeting metrics.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]

    def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = request.get("method")
        msg_id = request.get("id")
        params = request.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "resources": {}
                    },
                    "serverInfo": {
                        "name": "tok_n_compress",
                        "version": "2.0.0"
                    }
                }
            }

        elif method == "notifications/initialized":
            return None

        elif method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": self.tools}
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})
            try:
                result_text = self._execute_tool(tool_name, args)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": result_text}]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error executing {tool_name}: {str(e)}"}],
                        "isError": True
                    }
                }

        elif method == "resources/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "resources": [
                        {
                            "uri": "memory://checkpoint-map",
                            "name": "Episodic Checkpoint Map",
                            "mimeType": "application/json",
                            "description": "Hierarchical summary tree of active conversation checkpoints."
                        }
                    ]
                }
            }

        elif method == "resources/read":
            uri = params.get("uri")
            if uri == "memory://checkpoint-map":
                map_data = self.skill.get_checkpoint_map()
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": json.dumps(map_data, indent=2)
                            }
                        ]
                    }
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32602, "message": f"Unknown resource URI: {uri}"}
                }

        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Method '{method}' not found"
                }
            }

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        if name == "compress_conversation":
            res = self.skill.compress(
                raw_history=args.get("messages", []),
                user_query=args.get("user_query"),
                keep_recent_n=args.get("keep_recent_n", 2)
            )
            return json.dumps(res, indent=2)

        elif name == "query_memory":
            matches = self.skill.query(
                query_text=args.get("query", ""),
                limit=args.get("limit", 3),
                use_hybrid=args.get("use_hybrid", True)
            )
            return json.dumps(matches, indent=2)

        elif name == "rehydrate_segment":
            formatted = self.skill.rehydrate(
                target=args.get("segment_id"),
                user_query=args.get("user_query"),
                format_style=args.get("format_style", "xml")
            )
            return formatted

        elif name == "get_checkpoint_map":
            chk_map = self.skill.get_checkpoint_map(limit=args.get("limit", 50))
            return json.dumps(chk_map, indent=2)

        elif name == "pin_segment":
            ok = self.skill.pin_segment(
                segment_id=args.get("segment_id"),
                pinned=args.get("pinned", True)
            )
            return json.dumps({"segment_id": args.get("segment_id"), "pinned": args.get("pinned", True), "success": ok})

        elif name == "get_memory_stats":
            stats = self.skill.get_stats()
            return json.dumps(stats, indent=2)

        else:
            raise ValueError(f"Unknown tool: {name}")

    def run_stdio(self):
        """Standard input/output JSON-RPC line loop."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = self.handle_request(req)
                if resp is not None:
                    sys.stdout.write(json.dumps(resp) + "\n")
                    sys.stdout.flush()
            except Exception as e:
                logging.error(f"Error handling request: {e}")


def main():
    server = MCPServer()
    server.run_stdio()


if __name__ == "__main__":
    main()
