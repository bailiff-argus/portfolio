from bs4 import BeautifulSoup
import requests as rq
import pandas as pd
import numpy as np

"""

This program aims to pull data about the best restaurants in Moscow,
calculate their value -- the "bang for the buck" -- and return a dataframe
with restaurants sorted by value from highest to lowest.

Source: restoclub.ru

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
    try:
        return rest.findAll(
            lambda tag: tag.name == "li" and \
            tag.get('class') == ["search-place-card__info-item"]
        )[0].findAll('span')[0].text
    except IndexError: # no subway station nearby
        return None


def extract_and_adjust_score(rest) -> float:
    raw_score: float = float(rest.findAll(
        "div", attrs = {'class': "rating__value very-high"}
    )[0].text)

    # low amounts of reviews can introduce bias. To combat that, a 10.0
    # and a 1.0 reviews are added for any restaurant

    reviews: str = rest.findAll(
        "a", attrs = {'class': "search-place-rating__reviews"}
    )[0].text

    no_reviews: int = int(reviews.split(" ")[0])

    adjusted_score: float = round(
        (raw_score * no_reviews + 10.0 + 1.0) / (no_reviews + 2), 1
    )

    return adjusted_score


def extract_average_tab(rest) -> float:
    try:
        return float(rest.findAll(
            'li', attrs = {'class': "search-place-card__info-item _bill"}
        )[0].text)
    except IndexError:
        # encountered when the average tab isn't specified
        return np.nan


def parse_sublist(sublist) -> pd.DataFrame:
    rest_list = []

    rest_type = sublist.find("h3", attrs = {'class': "rc-rating__title"}).text
    rests = sublist.findAll("div", attrs = {'class': "search-place-card__description"})

    for i in range(len(rests)):
        rest_name: str = extract_name(rests[i])
        rest_subway_st: str = extract_subway_station(rests[i])
        rest_adjusted_score: float = extract_and_adjust_score(rests[i])
        rest_avg_tab = extract_average_tab(rests[i])

        # assuming food quality scales almost linearly with score,
        # how many score points per unit of money do you get?
        # since the score range is very tight, we'll adjust it for this
        # calculation
        rest_value: float = (rest_adjusted_score * 3 - 20) / rest_avg_tab

        rest_entry: dict = {
            'name': rest_name,
            'type': rest_type,
            'score': rest_adjusted_score,
            'average_tab': rest_avg_tab,
            'value': rest_value,
            'subway_st': rest_subway_st,
        }

        rest_list.append(rest_entry)

    rest_df = pd.DataFrame(rest_list)
    return rest_df



def main() -> None:
    top_restaurants = pd.DataFrame(columns = ('name', 'type', 'score', 'average_tab', 'value', 'subway_st'))

    raw_webpage_contents: bytes = pull_data_from_page()
    soup = BeautifulSoup(raw_webpage_contents, 'html.parser')

    rest_lists = soup.findAll('article', attrs = {'class': "rc-rating"})

    for j in range(len(rest_lists)):
        new_entry = parse_sublist(rest_lists[j])
        top_restaurants = pd.concat([top_restaurants, new_entry], ignore_index=True)

    # get rid of restaurants with data missing
    top_restaurants = top_restaurants.dropna()

    assert top_restaurants is not None

    # normalize the value scores, so that the highest is 100
    top_restaurants['value'] = round(top_restaurants['value'] * 100 / top_restaurants['value'].max(), 0)

    print(top_restaurants.sort_values("value", ascending=False).set_index("name"))


if __name__ == "__main__":
    main()
