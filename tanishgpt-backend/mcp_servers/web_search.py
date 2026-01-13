from duckduckgo_search import DDGS

def web_search(query: str, max_results: int = 5):
    """
    Performs a DuckDuckGo NEWS search and returns
    structured search results from the last week.
    """
    results = []

    try:
        with DDGS() as ddgs:
            # timelimit="w" -> last week, "d" -> last day
            for r in ddgs.news(query, max_results=max_results, timelimit="d"):
                results.append({
                    "title": r.get("title"),
                    "snippet": r.get("body"),
                    "url": r.get("url"),
                    "date": r.get("date")
                })
    except Exception as e:
        print(f"DDGS Error: {e}")
        return []

    return results
