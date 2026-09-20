from scrapling.fetchers import StealthyFetcher

URL = "https://mrpl.co.in/en/Content/Manufacturing_Units"

print("=" * 70)
print("MRPL + SCRAPLING TEST")
print("=" * 70)
print("\nOpening:", URL)

try:
    page = StealthyFetcher.fetch(
        URL,
        headless=False,
        network_idle=True,
        timeout=60000
    )

    print("\nSUCCESS")
    print("=" * 70)

    print("Status:", page.status)
    print("Final URL:", page.url)

    title = page.css("title::text").get()

    print("\nTITLE:")
    print(title)

    print("\nPAGE TEXT:")
    print("-" * 70)

    text = page.get_all_text()

    print(text[:10000])

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

except Exception as e:
    print("\nERROR")
    print("=" * 70)
    print(type(e).__name__)
    print(e)