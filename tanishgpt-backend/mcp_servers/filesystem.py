import os
import sys
import json
from pathlib import Path

# Configuration
WORKSPACE_DIR = Path(__file__).parent.parent / "agent_sketchpad"
WORKSPACE_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# TOOLS
# -------------------------------------------------
def write_file(relative_path: str, content: str) -> str:
    """
    Writes content to a file within the agent_workspace directory.
    
    Args:
        relative_path: The path to the file relative to the workspace root.
        content: The text content to write.
    """
    try:
        # Security: Prevent directory traversal
        if ".." in relative_path or relative_path.startswith("/"):
            return f"Error: Invalid path '{relative_path}'. Access denied."
            
        target_path = WORKSPACE_DIR / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return f"Successfully wrote to {relative_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def list_tools():
    return [
        {
            "name": "write_file",
            "description": "Write code or text to a file in the workspace. Use this to save your work.",
            "parameters": {
                "type": "object",
                "properties": {
                    "relative_path": {
                        "type": "string",
                        "description": "Relative path to the file (e.g., 'script.py' or 'src/main.py')"
                    },
                    "content": {
                        "type": "string",
                        "description": "The full content of the file"
                    }
                },
                "required": ["relative_path", "content"]
            }
        }
    ]

# -------------------------------------------------
# MCP SERVER LOOP (Simple Stdio)
# -------------------------------------------------
def run_server():
    """
    A minimal MCP-like server loop that reads JSON-RPC from stdin.
    """
    # Note: This is a simplified implementation for the "Hello World" phase.
    # In a real production MCP server, we would use the official SDK.
    # But this is enough to demonstrate the concept and work with our custom client.
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            
            request = json.loads(line)
            method = request.get("method")
            params = request.get("params", {})
            req_id = request.get("id")
            
            response = {"jsonrpc": "2.0", "id": req_id}
            
            if method == "tools/list":
                response["result"] = {"tools": list_tools()}
            
            elif method == "tools/call":
                tool_name = params.get("name")
                args = params.get("arguments", {})
                
                if tool_name == "write_file":
                    result = write_file(args.get("relative_path"), args.get("content"))
                    response["result"] = {"content": [{"type": "text", "text": result}]}
                else:
                    response["error"] = {"code": -32601, "message": "Method not found"}
            
            else:
                # Ignore other methods for now or send error
                continue
                
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
            
        except Exception as e:
            # Log error to stderr so it doesn't break JSON-RPC on stdout
            sys.stderr.write(f"Server Error: {e}\n")
            sys.stderr.flush()

if __name__ == "__main__":
    run_server()
