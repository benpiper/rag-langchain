import subprocess
import json
import sys
import os


def run_verification():
    # Path to the server script
    server_script = os.path.abspath("mcp_server.py")

    # Start the server process
    process = subprocess.Popen(
        [sys.executable, server_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=0,
    )

    try:
        # 1. Initialize
        # MCP requires an initialization handshake.
        # Client sends 'initialize' request.
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "verifier", "version": "0.1.0"},
            },
            "id": 1,
        }

        print("Sending initialize request...")
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()

        response = process.stdout.readline()
        print(f"Received: {response}")
        resp_json = json.loads(response)
        if "error" in resp_json:
            print(f"Error during initialization: {resp_json['error']}")
            return

        # 2. Initialized notification
        init_notif = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }
        process.stdin.write(json.dumps(init_notif) + "\n")
        process.stdin.flush()

        # 3. List Tools
        list_tools_req = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 2,
        }
        print("Sending tools/list request...")
        process.stdin.write(json.dumps(list_tools_req) + "\n")
        process.stdin.flush()

        response = process.stdout.readline()
        print(f"Received: {response}")
        resp_json = json.loads(response)
        tools = resp_json.get("result", {}).get("tools", [])
        found_tool = False
        for tool in tools:
            if tool["name"] == "query_knowledge_base":
                found_tool = True
                break

        if not found_tool:
            print("Error: query_knowledge_base tool not found.")
            return

        # 4. Call Tool
        call_tool_req = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "query_knowledge_base",
                "arguments": {"query": "evidence for the resurrection"},
            },
            "id": 3,
        }
        print("Sending tools/call request...")
        process.stdin.write(json.dumps(call_tool_req) + "\n")
        process.stdin.flush()

        response = process.stdout.readline()
        print(f"Received: {response}")
        resp_json = json.loads(response)

        content = resp_json.get("result", {}).get("content", [])
        if content:
            print("Verification Successful!")
            print(content[0].get("text", "")[:1000] + "...")
        else:
            print("Error: No content returned.")

    except Exception as e:
        print(f"Verification failed: {e}")
    finally:
        process.terminate()


if __name__ == "__main__":
    run_verification()
