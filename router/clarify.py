"""The one question to ask a customer whose request is too vague to route.

Routing never calls an API: tested on the same requests, a hosted model routed no better than the local
one (eval/llm_experiment.json). Where it can help is wording this question so it fits what the customer
actually wrote. That is optional. Without GROQ_API_KEY, or if the call fails, a fixed question is used and
the response says so. The service works the same either way.

Cost: only requests below the ask-first threshold reach this, about 16% of volume.
"""
import json
import os
import urllib.error
import urllib.request

ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.environ.get("KESTREL_QUESTION_MODEL", "openai/gpt-oss-120b")

TEMPLATE = ("Thanks for reaching out about your {product}. So we can send you to the right team straight away, "
            "is it about: a fault or breakdown, an installation or demo, filters or spare parts, a payment or "
            "invoice, a return or replacement, or warranty and Kestrel Shield?")

SYSTEM = """You write one short message to a Kestrel Home Appliances customer whose service request is too vague to \
route. Ask a single question that lets them say which of the likely needs applies. Match their tone and language \
(they may write in Hinglish). Under 45 words. No promises about timelines, refunds or visits. No numbers. Return \
JSON: {"question": "..."}"""


def template(product: str) -> str:
    return TEMPLATE.format(product=(product or "product").lower())


def ask(text: str, product: str, likely_teams: list, with_usage: bool = False) -> dict:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return {"question": template(product), "source": "standard question (no GROQ_API_KEY set)"}
    body = {"model": MODEL, "max_completion_tokens": 600, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": json.dumps({"customer_message": text, "product": product,
                                                                 "likely_needs": likely_teams})}]}
    if MODEL.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "kestrel-router/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            resp = json.loads(r.read())
        q = json.loads(resp["choices"][0]["message"]["content"])["question"].strip()
        if not q or len(q) > 400 or any(ch.isdigit() for ch in q):
            raise ValueError("question failed checks")
        out = {"question": q, "source": f"written for this message ({MODEL})"}
        if with_usage:
            out["usage"] = resp.get("usage", {})
        return out
    except urllib.error.HTTPError as e:
        reason = "API key rejected" if e.code in (401, 403) else f"API returned {e.code}"
    except (urllib.error.URLError, TimeoutError):
        reason = "API unreachable"
    except (KeyError, ValueError, TypeError):
        reason = "API reply unusable"
    return {"question": template(product), "source": f"standard question ({reason})"}
