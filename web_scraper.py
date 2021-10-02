from bs4 import BeautifulSoup
import requests as rq
import pandas as pd

"""

This script aims to pull names, nearest subway stations, scores, and types of the highest
ranked restaraunts in Moscow, source: restoclub.ru

"""

def pull_data_from_page() -> bytes:
    headers: dict = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36",
    }

    response = rq.get(
        url = "https://www.restoclub.ru/msk/ratings",
        headers = headers,
    )

    response.raise_for_status()

    return response.content



def extract_name(rest) -> str:
    return rest.findAll("span", attrs = {
        'class': "search-place-title__name",
    })[0].text


def extract_subway_station(rest) -> str:
    return rest.findAll(
        lambda tag: tag.name == "li" and \
        tag.get('class') == ["search-place-card__info-item"]
    )[0].findAll('span')[0].text


def extract_and_adjust_score(rest) -> float:
    raw_score: float = float(rest.findAll(
        "div", attrs = {'class': "rating__value very-high"}
    )[0].text)

    # low amounts of reviews can introduce bias. To combat that, a 10.0
    # and a 1.0 reviews are added for any restaraunt

    reviews: str = rest.findAll(
        "a", attrs = {'class': "search-place-rating__reviews"}
    )[0].text

    no_reviews: int = int(reviews.split(" ")[0])

    adjusted_score: float = round(
        (raw_score * no_reviews + 10.0 + 1.0) / (no_reviews + 2), 1
    )

    return adjusted_score


def parse_sublist(sublist) -> pd.DataFrame:
    rest_list = []

    rest_type = sublist.find("h3", attrs = {'class': "rc-rating__title"}).text
    rests = sublist.findAll("div", attrs = {'class': "search-place-card__description"})

    for i in range(len(rests)):
        rest_name: str = extract_name(rests[i])

        try:
            rest_subway_st: str = extract_subway_station(rests[i])
        except IndexError: # prevents an error for restaraunts with no subway stations stated
            continue

        rest_adjusted_score: float = extract_and_adjust_score(rests[i])

        rest_entry: dict = {
            'name': rest_name,
            'type': rest_type,
            'score': rest_adjusted_score,
            'subway_st': rest_subway_st,
        }

        rest_list.append(rest_entry)

    rest_df = pd.DataFrame(rest_list)
    return rest_df



def main() -> None:
    top_restaraunts = pd.DataFrame(columns = ("name", "type", "score", "subway_st"))

    raw_webpage_contents: bytes = pull_data_from_page()
    soup = BeautifulSoup(raw_webpage_contents, "html.parser")

    rest_lists = soup.findAll( "article", attrs = {'class': "rc-rating"})

    for j in range(len(rest_lists)):
        new_entry = parse_sublist(rest_lists[j])
        top_restaraunts = pd.concat([top_restaraunts, new_entry], ignore_index=True)

    print(top_restaraunts.sort_values("score", ascending=False))


if __name__ == "__main__":
    main()
