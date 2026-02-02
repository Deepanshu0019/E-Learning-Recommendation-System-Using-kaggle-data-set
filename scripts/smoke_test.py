import requests


BASE = "http://127.0.0.1:8000"


def check(path: str):
    url = f"{BASE}{path}"
    try:
        resp = requests.get(url, allow_redirects=False, timeout=5)
        status = resp.status_code
        loc = resp.headers.get("Location")
        print(f"{path}: {status}" + (f" -> {loc}" if loc else ""))
        return status
    except Exception as e:
        print(f"{path}: ERROR {e}")
        return None


def main():
    paths = [
        "/",
        "/courses",
        "/free",
        "/about",
        "/contact",
        "/search?q=python",
        "/recommendations",  # should 302 to login when anon
    ]
    for p in paths:
        check(p)


if __name__ == "__main__":
    main()


