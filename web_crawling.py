import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import pandas as pd
import time
import random
import json

cookies = {
    "bggusername": "TYY_1121",
    "SessionID": "dfafb3de46df7e10725dd95bc93661558f61cfe0u4182161"
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://boardgamegeek.com/",
    "Accept-Encoding": "gzip, deflate, br"
}

# Get the above information from the browser's developer tools
# Cookies: Application -> Cookies
# Headers: Network -> Request Headers
# You may need to refresh the page to see the request in the developer tools

def get_bgg_page(page_num, error_list):

    url = f"https://boardgamegeek.com/browse/boardgame/page/{page_num}"
    try:
        response = requests.get(url, headers=headers, cookies=cookies, timeout=10)
        response.raise_for_status()
        print("Getting page", page_num)
        # return response

        soup = BeautifulSoup(response.text, "html.parser")

        games = soup.select("tr[id^='row_']")  

        game_list = []
        for i, game in enumerate(games):

            rank = game.select_one(".collection_rank").text.strip()
            title_tag = game.select_one(".primary")
            title = title_tag.text.strip()
            game_id = int(title_tag["href"].split("/")[2])
            year = game.find("span", class_="smallerfont dull")
            year = year.text.strip() if year else None

            ratings = game.find_all("td", class_="collection_bggrating")
            geek_rating = ratings[0].text.strip()
            avg_rating = ratings[1].text.strip()
            
            link = "https://boardgamegeek.com" + title_tag["href"]

            game_list.append({
                "Rank": rank,
                "Name": title,
                "ID": game_id,
                "Year": year,
                "Geek Rating": geek_rating,
                "Avg Rating": avg_rating,
                "Link": link
            })

        for i in range(5):
            game_ids = ",".join([str(game["ID"]) for game in game_list[0+20*i:20+20*i]])
            get_game_info(game_ids, game_list, i)
            time.sleep(random.uniform(0.2, 0.8))


        # game_ids = ",".join([str(game["ID"]) for game in game_list])
        # get_game_info(game_ids, game_list)

        # time.sleep(random.uniform(1.2, 1.4))
            # return response

            
        
        return game_list
    
    except Exception as e:
        print(f"Page {page_num} Failed: {str(e)}")
        error_list.append(page_num)

        

def get_game_info(game_ids, gamelist, batch_id):
    api_url = f"https://api.geekdo.com/xmlapi2/thing?id={game_ids}&stats=1"
    response = requests.get(api_url, headers=headers, cookies=cookies)
    while response.status_code != 200:
        print(f"API ERROR at batch {batch_id}: {response.status_code}. Retrying...")

        time.sleep(random.uniform(3.3, 4.5))
        response = requests.get(api_url, headers=headers, cookies=cookies)

    xml_data = response.text

    if xml_data:
        # try:
        root = ET.fromstring(xml_data)
        for idx, game in enumerate(root.findall("item")):  # every <item>

            name = game.find("name").get("value")
            description = game.find("description").text

            min_players = game.find("minplayers").get("value")
            max_players = game.find("maxplayers").get("value")

            min_play_time = game.find("minplaytime").get("value")
            play_time = game.find("playingtime").get("value")

            weight = game.find("statistics/ratings/averageweight").get("value")


            categories = [link.get("value") for link in game.findall("link[@type='boardgamecategory']")]
            mechanics = [link.get("value") for link in game.findall("link[@type='boardgamemechanic']")]
            families = [link.get("value") for link in game.findall("link[@type='boardgamefamily']")]

            gamelist[batch_id*20+idx]["Min Players"] = min_players
            gamelist[batch_id*20+idx]["Max Players"] = max_players
            gamelist[batch_id*20+idx]["Min Play Time"] = min_play_time
            gamelist[batch_id*20+idx]["Play Time"] = play_time
            gamelist[batch_id*20+idx]["Weight"] = weight
            gamelist[batch_id*20+idx]["Description"] = description
            gamelist[batch_id*20+idx]["Categories"] = categories
            gamelist[batch_id*20+idx]["Mechanics"] = mechanics
            gamelist[batch_id*20+idx]["Families"] = families
        
        # except Exception as e:
        #     print(f"XML ERROR:", str(e))
        #     return None


        # print(f"Name: {name}")
        # print(f"Description: {description[:200]}...")
        # print(f"Categories: {', '.join(categories)}")
        # print(f"Players: {min_players} - {max_players}")
        # print(f"Time: {play_time} 分钟")
        # print(f"Complexity: {weight} / 5")
        # print(f"Mechanics: {', '.join(mechanics)}")
        # print(f"Families: {', '.join(families)}")

if __name__ == "__main__":
    error_list = []
    for i in range(1, 100):
        start_time = time.time()

        game_list = get_bgg_page(i, error_list)

        with open(f"data/bgg_data_{i}.json", "w") as f:
            json.dump(game_list, f, indent=2)

        print("Saving page", i, "at time", time.time() - start_time)

    print("Errors:", error_list)
    print("Done!")

