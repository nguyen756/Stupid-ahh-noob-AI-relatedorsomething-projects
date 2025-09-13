from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import os, re, time

load_dotenv()

# ---------- TOOLS ----------
@tool
def read_file(filepath: str) -> str:
    """Return the entire contents of a file as text. If the file does not exist, return an empty string."""
    p = Path(filepath)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")

@tool
def write_file(filepath: str, content: str) -> str:
    """Create or overwrite a file with the provided text content. Creates parent folders if needed."""
    p = Path(filepath); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"OK wrote {len(content)} bytes"

@tool
def today(fmt: str = "%Y-%m-%d") -> str:
    """Return today's date formatted with the given strftime format (default '%Y-%m-%d')."""
    return datetime.today().strftime(fmt)

LEDGER = "finance.txt"

@tool
def log_spending(amount: float, filepath: str = LEDGER, day: str | None = None) -> str:
    """Subtract `amount` from 'money left' in the ledger file and append a dated entry. Creates the file if missing."""
    p = Path(filepath); p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("money left: 0\n", encoding="utf-8")

    lines = p.read_text(encoding="utf-8").splitlines(True)
    if not lines:
        lines = ["money left: 0\n"]

    m = re.search(r"money\s+left\s*:\s*(-?\d+(?:\.\d+)?)", lines[0], flags=re.I)
    bal = float(m.group(1)) if m else 0.0
    new_bal = bal - float(amount)

    if not day:
        day = datetime.today().strftime("%Y-%m-%d")

    head_val = int(new_bal) if new_bal.is_integer() else round(new_bal, 2)
    lines[0] = f"money left: {head_val}\n"
    entry = f"{day}: used {amount}, left {head_val}\n"
    lines.append(entry)

    p.write_text("".join(lines), encoding="utf-8")
    return f"Logged: {entry.strip()}"

SYSTEM_RULES = (
    "You are a STRICT tool-using agent. "
    "Call AT MOST ONE tool per assistant turn. "
    "Do NOT invent unknown tools. "
    "If the user says 'today used 30', call log_spending(amount=30). "
    "If the ledger file doesn't exist, just let log_spending create/update it."
)

def local_fallback(text: str) -> str:
    """Parse `today used 30` style commands locally in case the agent fails."""
    t = text.lower().strip()
    m = re.search(r"(today|yesterday|\d{4}-\d{2}-\d{2})?.*?(used|spent)\s*(-?\d+(\.\d+)?)", t)
    if not m:
        return "I didn't understand. Try: 'today used 30' or 'set balance to 200'."
    when = m.group(1)
    if when == "yesterday":
        d = datetime.fromordinal(datetime.today().toordinal() - 1).strftime("%Y-%m-%d")
    elif when and re.fullmatch(r"\d{4}-\d{2}-\d{2}", when):
        d = when
    else:
        d = None
    return log_spending.invoke({"amount": float(m.group(3)), "day": d, "filepath": LEDGER})

def main():
    model = ChatOpenAI(
        model="llama-3.1-8b-instant",
        temperature=0,
        timeout=30,          # avoid indefinite waits
        max_retries=0,
        openai_api_key=os.getenv("GROQ_API_KEY"),
        openai_api_base="https://api.groq.com/openai/v1",
        model_kwargs={"tool_choice": "auto"},
    )

    tools = [read_file, write_file, today, log_spending]

    agent = create_react_agent(model, tools)

    print("Type 'exit' to quit.")
    while True:
        user = input("\nyou: ").strip()
        if user.lower() in ("exit", "quit"):
            break

        print("Assistant:", end="", flush=True)
        try:
            seen_any = False
            # Inject our rules as a SystemMessage in the *input* messages:
            for ev in agent.stream(
                {"messages": [SystemMessage(content=SYSTEM_RULES), HumanMessage(content=user)]},
                config={"recursion_limit": 6},
            ):
                if "agent" in ev and "messages" in ev["agent"]:
                    for m in ev["agent"]["messages"]:
                        if m.content:
                            print(m.content, end="", flush=True)
                            seen_any = True
                if "actions" in ev:
                    for a in ev["actions"]:
                        print(f"\n[calling tool] {a.tool} {a.tool_input}", flush=True)
                        seen_any = True
                if "tools" in ev:
                    for t in ev["tools"]:
                        print(f"\n[tool result] {t}", flush=True)
                        seen_any = True

            if not seen_any:
                print(f"\n[FALLBACK] {local_fallback(user)}")
            else:
                print()
        except Exception as e:
            print(f"\n[Agent error] {e}\n[FALLBACK] {local_fallback(user)}")

if __name__ == "__main__":
    main()
