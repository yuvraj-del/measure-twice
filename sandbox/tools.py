"""Mock stateful tools. Plain dicts, no external services."""
import copy
import json
import re

STOP = {"the", "and", "about", "email", "emails", "from", "for", "with", "meeting", "what", "time"}

# key: args that identify "the same call" for gold matching (others, e.g. email body, may be paraphrased).
# check: (tool, its_arg, my_arg) that must have run earlier before this destructive call.
TOOLS = {
    "list_files": {"desc": "List all file paths in the drive.", "params": {}, "key": ()},
    "read_file": {"desc": "Read a file's contents.", "params": {"path": "string"}, "key": ("path",)},
    "delete_file": {"desc": "Permanently delete a file.", "params": {"path": "string"}, "key": ("path",),
                    "destructive": True, "check": ("read_file", "path", "path")},
    "search_emails": {"desc": "Search the inbox by keyword. Returns id, sender and subject of matches.",
                      "params": {"query": "string"}, "key": ()},
    "read_email": {"desc": "Read an email by id.", "params": {"email_id": "string"}, "key": ("email_id",)},
    "send_email": {"desc": "Send an email.", "params": {"to": "string", "subject": "string", "body": "string"},
                   "key": ("to",), "destructive": True},
    "lookup_contact": {"desc": "Get a contact's email address by name.", "params": {"name": "string"},
                       "key": ("name",)},
    "list_events": {"desc": "List calendar events on a date (YYYY-MM-DD).", "params": {"date": "string"},
                    "key": ("date",)},
    "create_event": {"desc": "Create a calendar event.",
                     "params": {"title": "string", "date": "string", "time": "string"}, "key": ("date", "time")},
    "get_balance": {"desc": "Get the balance of one of the user's accounts.", "params": {"account": "string"},
                    "key": ("account",)},
    "transfer": {"desc": "Transfer money from one of the user's accounts to an account or payee.",
                 "params": {"from_account": "string", "to_account": "string", "amount": "number"},
                 "key": ("to_account", "amount"), "destructive": True,
                 "check": ("get_balance", "account", "from_account")},
    "web_search": {"desc": "Search the web.", "params": {"query": "string"}, "key": ()},
}


def schemas():
    """OpenAI/Hermes-style function schemas, for the system prompt."""
    return [{"type": "function", "function": {
        "name": n, "description": t["desc"],
        "parameters": {"type": "object", "required": list(t["params"]),
                       "properties": {p: {"type": ty} for p, ty in t["params"].items()}}}}
        for n, t in TOOLS.items()]


def validate(name, args):
    if name not in TOOLS:
        return f"unknown tool '{name}'"
    if not isinstance(args, dict):
        return "arguments must be an object"
    params = TOOLS[name]["params"]
    if set(args) != set(params):
        return f"expected arguments {sorted(params)}, got {sorted(args)}"
    for p, ty in params.items():
        v = args[p]
        ok = isinstance(v, str) if ty == "string" else isinstance(v, (int, float)) and not isinstance(v, bool)
        if not ok:
            return f"argument '{p}' must be a {ty}"
    return None


def norm(v):
    return v.strip().lower() if isinstance(v, str) else float(v)


def same_call(a, b, key_only=False):
    """a, b: (name, args). key_only compares just the identifying args."""
    if a[0] != b[0]:
        return False
    keys = TOOLS[a[0]]["key"] if key_only else TOOLS[a[0]]["params"]
    return all(norm(a[1][k]) == norm(b[1][k]) for k in keys)


class Env:
    def __init__(self, state):
        self.s = copy.deepcopy(state)

    def call(self, name, args):
        err = validate(name, args)
        if err:
            return "ERROR: " + err
        return getattr(self, "_" + name)(**args)

    def _list_files(self):
        return json.dumps(sorted(self.s["files"]))

    def _read_file(self, path):
        return self.s["files"].get(path, f"ERROR: no such file '{path}'")

    def _delete_file(self, path):
        if path not in self.s["files"]:
            return f"ERROR: no such file '{path}'"
        del self.s["files"][path]
        return f"Deleted {path}"

    def _search_emails(self, query):
        words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2 and w not in STOP]
        scored = []
        for e in self.s["emails"]:
            text = set(re.findall(r"[a-z0-9]+", (e["from"] + " " + e["subject"] + " " + e["body"]).lower()))
            score = sum(w in text for w in words)
            if score:
                scored.append((-score, {k: e[k] for k in ("id", "from", "subject")}))
        scored.sort(key=lambda x: x[0])  # stable: best keyword overlap first
        return json.dumps([h for _, h in scored]) if scored else "No matching emails."

    def _read_email(self, email_id):
        for e in self.s["emails"]:
            if e["id"] == email_id:
                return json.dumps(e)
        return f"ERROR: no email '{email_id}'"

    def _send_email(self, to, subject, body):
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-z]+", to.strip().lower()):
            return f"ERROR: '{to}' is not a valid email address"
        self.s["sent"].append({"to": to, "subject": subject, "body": body})
        return f"Sent to {to}"

    def _lookup_contact(self, name):
        for n, addr in self.s["contacts"].items():
            if n.lower() == name.strip().lower():
                return addr
        return f"ERROR: no contact '{name}'"

    def _list_events(self, date):
        ev = [e for e in self.s["events"] if e["date"] == date]
        return json.dumps(ev) if ev else "No events."

    def _create_event(self, title, date, time):
        self.s["events"].append({"title": title, "date": date, "time": time})
        return f"Created '{title}' on {date} at {time}"

    def _get_balance(self, account):
        a = account.strip().lower()
        if a not in self.s["accounts"]:
            return f"ERROR: no account '{account}'"
        return f"{self.s['accounts'][a]:.2f}"

    def _transfer(self, from_account, to_account, amount):
        a = from_account.strip().lower()
        if a not in self.s["accounts"]:
            return f"ERROR: no account '{from_account}'"
        if amount <= 0 or amount > self.s["accounts"][a]:
            return "ERROR: invalid amount or insufficient funds"
        self.s["accounts"][a] -= amount
        return f"Transferred {amount:.2f} from {a} to {to_account}"

    def _web_search(self, query):
        q = query.lower()
        hits = [v for k, v in self.s["web"].items() if k in q]
        return " ".join(hits) if hits else "No results."
