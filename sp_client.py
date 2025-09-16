# sp_client.py
import requests

class SPClient:
    def __init__(self, site_url: str, token: str):
        self.base = site_url.rstrip("/")
        self.s = requests.Session()
        self.s.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/json;odata=nometadata"
        })

    def _url(self, path: str):
        return f"{self.base}/_api/{path.lstrip('/')}"

    def get_pending(self, list_title: str, top: int = 200):
        """
        Reads up to `top` items with Status == 'Pending'
        Returns: list of dicts with fields we select.
        """
        select = "Id,MessageId,Subject,BodyPreview,From,Status"
        url = self._url(
            f"web/lists/getbytitle('{list_title}')/items"
            f"?$select={select}&$filter=Status eq 'Pending'&$top={top}"
        )
        r = self.s.get(url, timeout=30); r.raise_for_status()
        data = r.json()
        items = data.get("value", [])
        # Follow nextLink if present (rare for small batches)
        next_link = data.get("@odata.nextLink")
        while next_link and len(items) < top:
            r = self.s.get(next_link, timeout=30); r.raise_for_status()
            d = r.json()
            items.extend(d.get("value", []))
            next_link = d.get("@odata.nextLink")
        return items[:top]

    def update_item(self, list_title: str, item_id: int, payload: dict):
        """
        MERGE (PATCH) the list item with new fields.
        """
        url = self._url(f"web/lists/getbytitle('{list_title}')/items({item_id})")
        headers = {
            "Content-Type": "application/json;odata=nometadata",
            "IF-MATCH": "*",
            "X-HTTP-Method": "MERGE",
        }
        r = self.s.post(url, headers=headers, json=payload, timeout=30)
        if r.status_code not in (200, 204):
            # fallback to PATCH for tenants that block MERGE verb
            r = self.s.patch(url, headers={"Content-Type":"application/json;odata=nometadata","IF-MATCH":"*"},
                             json=payload, timeout=30)
        r.raise_for_status()
