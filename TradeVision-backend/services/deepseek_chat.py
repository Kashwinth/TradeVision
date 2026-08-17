"""
DeepSeek-backed chat fallback, using raw HTTP requests.

This module provides the exact same interface as gemini_chat.py but targets
the DeepSeek API (OpenAI-compatible) via the `requests` library.
"""

import json
import os
import requests
import threading

# We reuse the same configuration vars as ai_analysis.py
_API_URL = os.getenv("AI_ANALYSIS_API_URL", "https://api.deepseek.com/v1/chat/completions")
_API_KEY = os.getenv("AI_ANALYSIS_API_KEY", "")
_MODEL   = os.getenv("AI_ANALYSIS_MODEL", "deepseek-chat")

MAX_TOOL_ROUNDS = 5
_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
_TIMEOUT = int(os.getenv("AI_ANALYSIS_TIMEOUT", "60"))

_session_lock = threading.Lock()
_session: "requests.Session | None" = None

class DeepseekUnavailableError(RuntimeError):
    """No API key configured."""

class DeepseekChatError(RuntimeError):
    """The DeepSeek call itself failed."""

# Re-use the system prompt from gemini_chat
from services.gemini_chat import SYSTEM_PROMPT, compact

# Convert Gemini tool declarations to OpenAI/DeepSeek format
_OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_analysis",
            "description": (
                "Run TradeVision's full analysis for one CSE stock: XGBoost next-day "
                "direction with probability, the technical indicator summary, the "
                "as_of date of the newest real trading bar, and any data warnings. "
                "Use this for any question about a prediction, forecast, outlook, or "
                "whether a stock looks strong or weak."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "CSE ticker, e.g. JKH.N0000. Short form 'JKH' is accepted.",
                    },
                    "include_news": {
                        "type": "boolean",
                        "description": (
                            "ALWAYS pass true to ensure the FinBERT sentiment score is "
                            "fetched and included in the response."
                        ),
                    },
                },
                "required": ["symbol"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_quote",
            "description": (
                "Live CSE quote for one stock: last traded price, change, change "
                "percent, volume, turnover, day high/low/open, previous close and "
                "market cap. This is the current market price — use it for "
                "'what is X trading at' questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "CSE ticker, e.g. JKH.N0000. Short form 'JKH' is accepted.",
                    },
                },
                "required": ["symbol"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_gainers",
            "description": "Today's biggest percentage gainers on the CSE (up to 10).",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_losers",
            "description": "Today's biggest percentage losers on the CSE (up to 10).",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_most_active",
            "description": "Today's most actively traded CSE counters by share volume (up to 10).",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_overview",
            "description": (
                "Exchange-level state: whether the market is open or closed, the ASPI "
                "index level and its daily change, total turnover and share volume for "
                "the session, and the number of listed companies."
            ),
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_symbols",
            "description": (
                "Every company listed on the CSE, as symbol and name only. Use this to "
                "find a ticker when the user names a company instead of a symbol."
            ),
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_symbols",
            "description": (
                "Search for a CSE ticker symbol by company name. Use this to find the "
                "correct symbol when the user asks about a company (e.g. 'Hayleys' or "
                "'Sampath Bank') before calling other tools."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Company name or part of it to search for, e.g. 'Sampath'.",
                    },
                },
                "required": ["query"],
            }
        }
    },
]

_DECLARED_NAMES = {d["function"]["name"] for d in _OPENAI_TOOLS}

def is_available() -> bool:
    return bool(_API_URL.strip()) and bool(_API_KEY.strip())

def _get_session() -> "requests.Session":
    global _session
    if not is_available():
        raise DeepseekUnavailableError(
            "AI chat fallback is not configured: AI_ANALYSIS_API_KEY is not set."
        )

    with _session_lock:
        if _session is None:
            _session = requests.Session()
            _session.headers.update({
                "Content-Type": "application/json",
                "Authorization": f"Bearer {_API_KEY}",
            })
    return _session

def _to_openai_messages(messages: list[dict], system_prompt: str) -> list[dict]:
    contents = [{"role": "system", "content": system_prompt}]
    for message in messages:
        text = str(message.get("content") or "").strip()
        if not text:
            continue
        role = "user" if message.get("role") == "user" else "assistant"
        contents.append({"role": role, "content": text})
    return contents

def _run_tool(name: str, args: dict, handlers: dict) -> dict:
    handler = handlers.get(name)
    if handler is None:
        return {"error": f"Tool '{name}' is not available on this server."}
    try:
        result = handler(**args)
    except TypeError as e:
        return {"error": f"Invalid arguments for {name}: {e}"}
    except Exception as e:
        return {"error": f"{name} failed: {e}"}
    
    return result if isinstance(result, dict) else {"result": result}

def _describe(name: str, args: dict) -> str:
    if not args:
        return f"{name}()"
    inner = ", ".join(f"{k}={v}" for k, v in args.items())
    return f"{name}({inner})"

def chat(messages: list[dict], handlers: dict, symbol: str | None = None) -> dict:
    if not messages:
        raise DeepseekChatError("No messages to respond to.")
        
    missing = _DECLARED_NAMES - set(handlers)
    if missing:
        raise DeepseekChatError(f"Tool handlers missing for: {', '.join(sorted(missing))}")
        
    session = _get_session()
    
    system_prompt = SYSTEM_PROMPT
    if symbol:
        system_prompt += (
            f"\nThe user is currently viewing {symbol} in TradeVision. Resolve "
            f'bare references like "this stock" or "it" to {symbol}.\n'
        )
        
    contents = _to_openai_messages(messages, system_prompt)
    
    tools_used: list[str] = []
    warnings: list[str] = []
    
    for _ in range(MAX_TOOL_ROUNDS):
        payload = {
            "model": _MODEL,
            "messages": contents,
            "temperature": _TEMPERATURE,
            "tools": _OPENAI_TOOLS,
            "tool_choice": "auto"
        }
        
        try:
            resp = session.post(_API_URL, json=payload, timeout=_TIMEOUT)
            resp.raise_for_status()
            body = resp.json()
        except Exception as e:
            raise DeepseekChatError(f"DeepSeek request failed: {e}")
            
        message = body["choices"][0]["message"]
        
        # Tool call
        if message.get("tool_calls"):
            # Append the model's message exactly as it is (it contains tool_calls)
            contents.append(message)
            
            for call in message["tool_calls"]:
                name = call["function"]["name"]
                args_str = call["function"]["arguments"]
                try:
                    args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    args = {}
                    
                tools_used.append(_describe(name, args))
                result = _run_tool(name, args, handlers)
                
                if "error" in result:
                    warnings.append(str(result["error"]))
                    
                # OpenAI format for tool response
                contents.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": json.dumps(result)
                })
        else:
            # Done, we have text
            reply = message.get("content", "").strip()
            if not reply:
                raise DeepseekChatError("DeepSeek returned an empty response.")
            return {"reply": reply, "tools_used": tools_used, "warnings": warnings}
            
    # Force an answer without tools
    warnings.append(f"Stopped after {MAX_TOOL_ROUNDS} data lookups; answering from what was gathered.")
    payload = {
        "model": _MODEL,
        "messages": contents,
        "temperature": _TEMPERATURE,
    }
    
    try:
        resp = session.post(_API_URL, json=payload, timeout=_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        reply = body["choices"][0]["message"].get("content", "").strip()
    except Exception as e:
        raise DeepseekChatError(f"DeepSeek request failed: {e}")
        
    if not reply:
        raise DeepseekChatError(f"DeepSeek kept requesting data after {MAX_TOOL_ROUNDS} rounds without answering.")
        
    return {"reply": reply, "tools_used": tools_used, "warnings": warnings}
