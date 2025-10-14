
#most updated:
# import requests

# class SPClient:
#     def __init__(self, site_url: str, email: str, password: str):
#         self.base = site_url.rstrip("/")
#         self.session = requests.Session()
#         self.session.auth = (email, password)
#         self.session.headers.update({
#             "Accept": "application/json;odata=nometadata",
#             "Content-Type": "application/json"
#         })

#     def _url(self, path: str):
#         return f"{self.base}/_api/{path.lstrip('/')}"

#     def get_pending(self, list_title: str, top: int = 100):
#         select_fields = "Id,Title,Body,Sender,CC,MessageID,Status"
#         url = self._url(
#             f"web/lists/getbytitle('{list_title}')/items"
#             f"?$select={select_fields}&$filter=Status eq 'Pending'&$top={top}"
#         )
#         response = self.session.get(url, timeout=30)
#         response.raise_for_status()
#         return response.json().get("value", [])

#     def update_item(self, list_title: str, item_id: int, payload: dict):
#         url = self._url(f"web/lists/getbytitle('{list_title}')/items({item_id})")
#         headers = {
#             "IF-MATCH": "*",
#             "X-HTTP-Method": "MERGE"
#         }
#         response = self.session.post(url, headers=headers, json=payload, timeout=30)
#         if response.status_code not in (200, 204):
#             response = self.session.patch(url, headers={"IF-MATCH": "*"}, json=payload, timeout=30)
#         response.raise_for_status()

#     def create_item(self, list_title: str, payload: dict):
#         url = self._url(f"web/lists/getbytitle('{list_title}')/items")
#         response = self.session.post(url, json=payload, timeout=30)
#         response.raise_for_status()
#         return response.json()

#Refactored
# import os
# from office365.sharepoint.client_context import ClientContext
# from office365.runtime.auth.user_credential import UserCredential

# class SPClient:
#     def __init__(self):
#         site_url = os.environ.get("SHAREPOINT_SITE")
#         email = os.environ.get("SHAREPOINT_EMAIL")
#         password = os.environ.get("SHAREPOINT_PASSWORD")
#         self.ctx = ClientContext(site_url).with_credentials(UserCredential(email, password))

#     def get_pending(self, list_title: str, top: int = 100):
#         sp_list = self.ctx.web.lists.get_by_title(list_title)
#         items = sp_list.items.filter("Status eq 'Pending'").top(top).get().execute_query()
#         return [item.properties for item in items]

#     def update_item(self, list_title: str, item_id: int, payload: dict):
#         sp_list = self.ctx.web.lists.get_by_title(list_title)
#         item = sp_list.items.get_by_id(item_id)
#         for k, v in payload.items():
#             item.set_property(k, v)
#         item.update().execute_query()

#     def create_item(self, list_title: str, payload: dict):
#         sp_list = self.ctx.web.lists.get_by_title(list_title)
#         item = sp_list.add_item(payload).execute_query()
#         return item.properties

