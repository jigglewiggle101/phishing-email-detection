# sp_token.py
import os, msal

def get_sp_token():
    """
    Returns a bearer token scoped to your SharePoint tenant.
    Required env vars:
      SP_TENANT_ID, SP_CLIENT_ID, SP_CLIENT_SECRET, SP_SITE
    Optional:
      SP_SCOPE (override)  e.g. https://contoso.sharepoint.com/.default
    """
    tenant = os.environ["SP_TENANT_ID"]
    client_id = os.environ["SP_CLIENT_ID"]
    secret = os.environ["SP_CLIENT_SECRET"]
    site_url = os.environ["SP_SITE"].rstrip("/")

    # root like https://contoso.sharepoint.com
    root = site_url.split("/sites/")[0]
    scope = os.environ.get("SP_SCOPE", f"{root}/.default")

    app = msal.ConfidentialClientApplication(
        client_id,
        authority=f"https://login.microsoftonline.com/{tenant}",
        client_credential=secret,
    )
    result = app.acquire_token_for_client(scopes=[scope])
    if "access_token" not in result:
        raise RuntimeError(f"Failed to get token: {result}")
    return result["access_token"]
