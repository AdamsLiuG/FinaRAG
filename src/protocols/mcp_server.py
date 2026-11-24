"""MCP 2025-03-26 JSON-RPC bindings for stdio and stateless Streamable HTTP."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response

from src.storage.base import validate_safe_path

from .tool_registry import (
    MAX_ARGUMENT_BYTES,
    MCPToolRegistry,
    ToolPrincipal,
    build_financial_tool_registry,
    canonical_json,
    decode_json,
    default_policy,
)

PROTOCOL_VERSION = "2025-03-26"
RESOURCES_ROOT = Path(__file__).parent / "resources"
_MAX_FRAME_BYTES = MAX_ARGUMENT_BYTES + 4096
_MAX_BATCH_SIZE = 12
_MAX_STDIO_REQUESTS = 65536
_RESOURCES = {
    "finarag://schemas/config": ("config.schema.json", "Public FinaRAG configuration schema"),
    "finarag://schemas/report": ("report.schema.json", "Financial report metadata schema"),
}
_RPC_MESSAGES = {
    -32700: "Parse error",
    -32600: "Invalid Request",
    -32601: "Method not found",
    -32602: "Invalid params",
    -32603: "Internal error",
    -32002: "Server initialization required",
}
_TOOL_MESSAGES = {
    "invalid_arguments": "Invalid tool arguments",
    "forbidden": "Tool access denied",
    "not_found": "Tool or record not found",
    "unavailable": "Tool unavailable",
    "busy": "Tool executor busy",
    "timeout": "Tool deadline exceeded",
    "internal_error": "Tool execution failed",
}


def rpc_error(code, request_id=None, *, message=None):
    return {"jsonrpc": "2.0", "id": request_id,
            "error": {"code": code, "message": message or _RPC_MESSAGES[code]}}


def rpc_result(request_id, result):
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _valid_id(value):
    return (type(value) is int and -(2**63) <= value < 2**63
            or isinstance(value, str) and len(value) <= 128)


def valid_rpc_request(message):
    return (isinstance(message, dict)
            and not (message.keys() - {"jsonrpc", "id", "method", "params"})
            and message.get("jsonrpc") == "2.0"
            and isinstance(message.get("method"), str) and 0 < len(message["method"]) <= 128
            and ("id" not in message or _valid_id(message["id"])))


def _valid_initialize(params):
    required = {"protocolVersion", "capabilities", "clientInfo"}
    if not required <= params.keys() or params.keys() - required - {"_meta"}:
        return False
    info = params["clientInfo"]
    return (isinstance(params["protocolVersion"], str) and 0 < len(params["protocolVersion"]) <= 32
            and isinstance(params["capabilities"], dict)
            and isinstance(info, dict) and {"name", "version"} <= info.keys()
            and not (info.keys() - {"name", "title", "version"})
            and all(isinstance(value, str) and 0 < len(value) <= 128 for value in info.values()))


class MCPServer:
    def __init__(self, registry: MCPToolRegistry, *, resources_root):
        self.registry = registry
        self._resource_text = {}
        root = Path(resources_root).absolute()
        for uri, (filename, _) in _RESOURCES.items():
            path = validate_safe_path(root / filename, root)
            with path.open("rb") as source:
                schema = decode_json(source.read(MAX_ARGUMENT_BYTES + 1))
            if not isinstance(schema, dict) or schema.get("type") != "object":
                raise ValueError("Invalid protocol resource schema")
            self._resource_text[uri] = canonical_json(schema)

    def close(self):
        self.registry.close()

    async def dispatch(self, message, *, principal) -> dict | list | None:
        return await self._dispatch_frame(message, principal, stdio=None)

    async def _dispatch_frame(self, message, principal, stdio):
        if not isinstance(message, list):
            return await self._dispatch_one(message, principal, stdio)
        if not 1 <= len(message) <= _MAX_BATCH_SIZE:
            return rpc_error(-32600)
        ids = [entry["id"] for entry in message if valid_rpc_request(entry) and "id" in entry]
        if len(ids) != len(set(ids)):
            return rpc_error(-32600)
        replies = []
        for entry in message:
            reply = await self._dispatch_one(entry, principal, stdio)
            if reply is not None:
                replies.append(reply)
        return replies or None

    async def _dispatch_one(self, message, principal, stdio):
        if not valid_rpc_request(message):
            return rpc_error(-32600)
        method, request_id = message["method"], message.get("id")
        params = message.get("params", {})
        if request_id is None:
            if method == "notifications/initialized" and params == {} and stdio is not None:
                if stdio.state == "negotiated":
                    stdio.state = "ready"
            return None
        if stdio is not None:
            if request_id in stdio.request_ids or len(stdio.request_ids) >= _MAX_STDIO_REQUESTS:
                return rpc_error(-32600, request_id)
            stdio.request_ids.add(request_id)
        if not isinstance(params, (dict, list)):
            return rpc_error(-32602, request_id)
        if isinstance(params, dict) and "_meta" in params:
            if not isinstance(params["_meta"], dict):
                return rpc_error(-32602, request_id)
            params = {key: value for key, value in params.items() if key != "_meta"}
        if method == "initialize":
            if not isinstance(params, dict) or not _valid_initialize(params):
                return rpc_error(-32602, request_id)
            if stdio is not None:
                if stdio.state != "new":
                    return rpc_error(-32600, request_id)
                stdio.state = "negotiated"
            return rpc_result(request_id, {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}, "resources": {}},
                "serverInfo": {"name": "FinaRAG", "version": "1.0.0"},
            })
        if method == "ping":
            return rpc_result(request_id, {}) if params == {} else rpc_error(-32602, request_id)
        if stdio is not None and stdio.state != "ready":
            return rpc_error(-32002, request_id)
        if method == "tools/list" and params == {}:
            return rpc_result(request_id, {"tools": self.registry.list_tools(principal)})
        if method == "resources/list" and params == {}:
            resources = [
                {"uri": uri, "name": name, "mimeType": "application/schema+json"}
                for uri, (_, name) in _RESOURCES.items()
                if self.registry.allows(principal, method, {"uri": uri})
            ]
            return rpc_result(request_id, {"resources": resources})
        if method == "resources/read" and isinstance(params, dict) and set(params) == {"uri"}:
            uri = params["uri"]
            if (isinstance(uri, str) and uri in self._resource_text
                    and self.registry.allows(principal, method, {"uri": uri})):
                return rpc_result(request_id, {"contents": [
                    {"uri": uri, "mimeType": "application/schema+json", "text": self._resource_text[uri]},
                ]})
        if method == "tools/call" and isinstance(params, dict) and not params.keys() - {"name", "arguments"}:
            name, arguments = params.get("name"), params.get("arguments", {})
            if isinstance(name, str) and 0 < len(name) <= 100 and isinstance(arguments, dict):
                execution = await self.registry.call(name, arguments, principal=principal)
                result = {"content": []}
                if execution.error_code:
                    result["isError"] = True
                    content = canonical_json({"code": execution.error_code,
                                              "message": _TOOL_MESSAGES[execution.error_code]})
                else:
                    content = canonical_json(execution.data)
                result["content"].append({"type": "text", "text": content})
                return rpc_result(request_id, result)
        if method in {"tools/list", "tools/call", "resources/list", "resources/read"}:
            return rpc_error(-32602, request_id)
        return rpc_error(-32601, request_id)


def accepts_json(accept):
    if not accept:
        return True
    qualities = {}
    for item in accept.lower().split(","):
        media, *params = item.strip().split(";")
        media = media.strip()
        if media not in {"application/json", "application/*", "*/*"}:
            continue
        quality = 1.0
        for param in params:
            key, _, value = param.strip().partition("=")
            if key == "q":
                try:
                    quality = float(value)
                except ValueError:
                    quality = 0.0
        qualities[media] = max(quality, qualities.get(media, 0))
    for media in ("application/json", "application/*", "*/*"):
        if media in qualities:
            return 0 < qualities[media] <= 1
    return False


async def bounded_body(request, limit):
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > limit:
            raise HTTPException(status_code=413, detail="Request too large")
        data.extend(chunk)
    return bytes(data)


def mount_mcp_routes(app, server_provider, *, auth, allowed_origins):
    origins = frozenset(allowed_origins)

    @app.post("/mcp")
    async def mcp_post(request: Request):
        origin = request.headers.getlist("origin")
        if origin and (len(origin) != 1 or origin[0] not in origins):
            raise HTTPException(status_code=403, detail="Origin denied")
        principal = await auth(request)
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise HTTPException(status_code=415, detail="Expected application/json")
        if not accepts_json(request.headers.get("accept")):
            raise HTTPException(status_code=406, detail="JSON response is required")
        versions = request.headers.getlist("MCP-Protocol-Version")
        if versions and versions != [PROTOCOL_VERSION]:
            raise HTTPException(status_code=400, detail="Unsupported MCP protocol version")
        body = await bounded_body(request, _MAX_FRAME_BYTES)
        try:
            message = decode_json(body, limit=_MAX_FRAME_BYTES)
        except (ValueError, RecursionError):
            return JSONResponse(rpc_error(-32700), status_code=400)
        try:
            server = await asyncio.to_thread(server_provider)
        except Exception:
            raise HTTPException(status_code=503, detail="MCP unavailable") from None
        reply = await server.dispatch(message, principal=principal)
        if reply is None:
            return Response(status_code=202)
        return Response(canonical_json(reply), media_type="application/json")


class _StdioSession:
    def __init__(self):
        self.state = "new"
        self.request_ids = set()


async def _serve_stdio(server, source, sink):
    principal = ToolPrincipal("finarag-stdio", frozenset({"read"}))
    session = _StdioSession()
    while frame := source.readline(_MAX_FRAME_BYTES + 2):
        if len(frame) > _MAX_FRAME_BYTES + int(frame.endswith(b"\n")):
            while frame and not frame.endswith(b"\n"):
                frame = source.readline(_MAX_FRAME_BYTES + 2)
            reply = rpc_error(-32700)
        else:
            try:
                message = decode_json(frame.removesuffix(b"\n"), limit=_MAX_FRAME_BYTES)
            except (ValueError, RecursionError):
                reply = rpc_error(-32700)
            else:
                reply = await server._dispatch_frame(message, principal, session)
        if reply is not None:
            sink.write((canonical_json(reply) + "\n").encode("utf-8"))
            sink.flush()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Serve FinaRAG MCP over stdio")
    parser.add_argument("--corpus-root", default=os.getenv("FINARAG_PROTOCOL_CORPUS_ROOT"))
    args = parser.parse_args(argv)
    sink = sys.stdout.buffer
    registry = None
    with redirect_stdout(sys.stderr):
        try:
            if not args.corpus_root:
                raise ValueError("Required corpus root")
            registry = build_financial_tool_registry(args.corpus_root, policy=default_policy)
            server = MCPServer(registry, resources_root=RESOURCES_ROOT)
        except Exception:
            if registry is not None:
                registry.close()
            print("MCP configuration unavailable", file=sys.stderr)
            return 2
        try:
            asyncio.run(_serve_stdio(server, sys.stdin.buffer, sink))
        finally:
            server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
