# src/features_extra.py
import re, tldextract
from urllib.parse import urlparse
from bs4 import BeautifulSoup

URL_RE = re.compile(r"https?://[^\s)>'\"}]+", re.I)
SUSPICIOUS = [
  "urgent","verify","suspend","account locked","password","docu sign","docusign",
  "sharepoint","one drive","payment","invoice","gift card","click here","confirm identity"
]

def _urls(text): return URL_RE.findall(text or "")

def _domain(url: str):
    try:
        netloc = urlparse(url).netloc
        ext = tldextract.extract(netloc)
        return ".".join([p for p in [ext.domain, ext.suffix] if p])
    except Exception:
        return ""

def _html_sig(text: str):
    soup = BeautifulSoup(text or "", "html.parser")
    return {
        "n_links": len(soup.find_all("a")),
        "n_imgs": len(soup.find_all("img")),
        "has_form": int(bool(soup.find("form")))
    }

def make_signals(subject: str, body: str, sender: str = "", internal_domain: str | None = None):
    s = (subject or "").lower()
    b = (body or "").lower()
    urls = _urls(body)
    url_domains = [_domain(u) for u in urls if u]
    is_internal = int(bool(internal_domain) and (sender or "").lower().endswith(f"@{internal_domain}"))

    sig = {
        "n_urls": len(urls),
        "uniq_url_domains": len(set([d for d in url_domains if d])),
        "n_exclam": b.count("!"),
        "caps_in_subject": sum(1 for c in subject or "" if c.isalpha() and c.isupper()),
        "has_suspicious_phrase": int(any(p in s or p in b for p in SUSPICIOUS)),
        "is_internal_sender": is_internal,
        "internal_with_links": int(is_internal and len(urls) > 0),
    }
    sig.update(_html_sig(body))
    return sig
