Football Predictions Capstone Project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This capstone project consists of developing an injury predictor that can assist football managers or clubs to make a decision when it comes to investing in football players.

**The main features** of this library are:

- A high-level API allows data scraping from the FBRef website (https://fbref.com/) to obtain match logs data of signed players from the top 5 European Leagues (Spain, Italy, France, Germani, and England).
- A high-level API allows data scraping from the TransfMarkt website (https://www.transfermarkt.com/) to obtain all possible players' injury data.
- Reference Table of mapped IDs from Fbref players and TransferMarkt sites
- Time series ML models to build injury predictors ...


**Important note**

    Data scrapping from the FBRef website comes at a high computational cost since each of the top 5 European leagues has about 20 teams, and those teams have an 
    approximate number of 30+ players. The entire data set takes accounts of every single available season.
    
    Data scrapping from the FBRef website comes at a high computational cost since each of the top 5 European leagues has about 20 teams, and those teams have an approximate number of 30+ players.
    The entire data pulling accounts for every single season available.
    

    The Beautiful Soup Python library was used for pulling data from the web. This requires basic knowledge of how to read and interpret HTML and XML files.

    The following libraries need to be installed:

.. code:: python
    
    !pip install beautifulsoup4
    !pip install pyjsparser
    !pip install js2xml

Table of Contents
~~~~~~~~~~~~~~~~~
 - `Overview`_
 - `Quick Start`_
 - `Simple training pipeline`_
 - `Examples`_
 - `Models and Backbones`_
 - `Installation`_
 - `Documentation`_
 - `Change log`_
 - `Citing`_
 - `License`_
 
Overview
~~~~~~~~
- The main programming language used in this project is Python. 
- VSCode is the code-editor employed since it allows the connection of the GitHub repository as well as working cooperatively in real-time.
- Jupyter notebooks is the interface used to write, read and produce all scripts for data scrapping, manipulation, visualizations, and creation of 
  all Machine Learning models. 
- Google Drive has been mirrored into our local machines in order to read and write large files through VSCode since our GitHub repository had a 
  limited capacity. 
- An Amazon Web Services (AWS) environment had been generated and linked to VSCode in order to increase computational power and be more productive 
  when building applications that come at a high cost; our local machines experienced multiple memory timeouts and limitations.

Quick Start
~~~~~~~~~~~

First of all, import the following Libraries:

.. code:: python

    # Base
    import pandas as pd
    import numpy as np
    import pickle
    import re 

    # Visualization
    import plotly.express as ex

    # Web Scraping
    import pprint
    import requests
    from bs4 import BeautifulSoup
    from pyjsparser import parse
    import pyjsparser
    from urllib.request import urlopen

    # Time 
    import sys
    import time
    from datetime import date
    from datetime import datetime
    from termcolor import colored

    # GC
    import gc

    # Itertools
    import itertools

    # Grafikten Data Çekmek için
    import re
    import js2xml
    from itertools import repeat    
    from pprint import pprint as pp

    # Configurations
    import warnings
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pd.set_option('display.max_columns', None)


**1. FBREF Extract.ipynb**

In this notebook, we create an extensive list of all match logs for all players and all the seasons they played from the FBRef website. 
This also includes match logs of other competitions such as their previous clubs(even if they played outside of the top 5 leagues) as well as 
their national team matches. 


Use BeautifulSoup to first obtain the league URLs

.. code:: python

    # Big 5 European Leagues (Spain, England, Germany, France, Italy)

    big_5_leagues = []

    for j in soup.find_all('tbody')[2].find_all("tr", {"class": "gender-m"}):
        if (j.find('td') != None):
            big_5_leagues.append(j.find('a')['href'])

    big_5_leagues = big_5_leagues[:-1]

    # function to obtain league/season URLs

    def get_all_seasons(url):
        URL = 'https://fbref.com/' + url
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
        url_list = []
        
        for row in soup.find_all('tr'):
            if row.find('th',{"scope":"row"}) != None:
                url_list.append((row.find('a')['href']))
        
    return url_list

    # All Seasons Big 5 Leagues

    all_seasons_big_5 = []

    for i in big_5_leagues:
        league_seasons = get_all_seasons(i)
        all_seasons_big_5 += league_seasons

Pull all players' stats for all competitions to end up with a list of all players' URLs for every season they played. Please note that there are more 
steps during the data scrapping, but only the most important ones are shown; refer to the notebooks for the complete code.

.. code:: python

    # function to obtain matchlogs
    
    def get_players_all_competitions(player_list):
        
        player_urls = []

        for i in player_list:
            player_urls.append('https://fbref.com/en/players/' + i.split('/')[3:4][0] + '/all_comps/' 
                                + i.split('/')[7:][0].replace("-Match-Logs", "") + '/-Stats---All-Competitions')

        return list(set(player_urls))

    player_all_competitions = get_players_all_competitions(player_table_big_5)

The following function had to be applied in multiple batches since this operation required high computation; this method allowed us to produce a single list of 
all players after concatenating all the lists. Thus, a total of 4 batches of 5000 URLs were created to generate the **match_logs_urls list**.

.. code:: python

    # Generate the match log urls for all players across all leagues and seasons

    def get_player_match_logs(player_list_summary, line):
        
        res = requests.get(player_list_summary[line])
        soup = BeautifulSoup(res.text,'lxml')

        match_logs_list = []

        for i in soup.find_all('tbody'):
            for j in i.find_all('td', {'data-stat':'matches'}):
                if j.find('a') != None:
                    if 'summary' in j.find('a')['href']:
                        match_logs_list.append(j.find('a')['href'])
                        
        return list(set(match_logs_list))

    match_logs_list = []

    # 1st batch 0:5000 
    count = 0
    for i in range(len(player_all_competitions[0:5000])):
        match_logs_list.extend(get_player_match_logs(player_all_competitions[0:5000], i))
        count += 1
        sys.stdout.write("\r{0} percent".format((count / len(player_all_competitions[0:5000])*100)))
        sys.stdout.flush()

**1.5 Append match_url_files.ipynb**

In this notebook, we concatenate the match logs lists that were created above to build the final **match_log_urls** list that contains 
all players' URLs match logs for every single season. This list has 148,478 URLs

.. code:: python

    # Uniting all match logs into a single list: match_logs_list_urls

    match_logs_list_urls = []
    match_logs_list_urls.extend(list(match_logs_list_urls_1['0']))
    match_logs_list_urls.extend(list(match_logs_list_urls_2['0']))
    match_logs_list_urls.extend(list(match_logs_list_urls_3['0']))
    match_logs_list_urls.extend(list(match_logs_list_urls_4['0']))
    match_logs_list_urls.extend(list(match_logs_list_urls_5['0']))

However, we have to ensure this list contains unique URLs since some players appear in more than one of the top 5 European leagues in their careers. 
The final list reduced to 118,283 URLs. Finally, this list is exported into a CSV file since it the easiest and fastest methods to save file to 
the Google Drive.

.. code:: python

    # Eliminated Repeated match logs
    match_logs_list_urls = list(set(match_logs_list_urls))

    # Export as CSV
    pd.DataFrame(match_logs_list_urls).to_csv('/Volumes/GoogleDrive/......./CSV Files/match_logs_list_urls.csv')

**2. FBREF Player Batch 0-5000.ipynb, 3.FBREF Player Batch 0-5000.ipynB, ........., 13c. FBREF Player Batch 110000-118283** 

It is time to perform the real data scrapping. Here, we are pulling data from the above list, which contains a total of 118,283 URLs. 
By running this function, we are extracting the match logs of all seasons for every single player. In addition, we found that some players 
have match logs that contain 30 attributes or columns while other players have match logs with 39 attributes. Thus, players' match logs are 
appended to two dataframes of 30 columns and 39 columns, respectively. 

**Important note**

    This step took a significant amount of memory usage. Therefore, it was necessary to run the match_logs_list_urls.csv in multiple batches. 
    A total of 15 notebooks were created in order to run all batches in parallel. The function below is used across all FBREF Player Batch notebooks; 
    this is an example of the first batch. At the end, all dataframes will be concatenated together to produce a single dataframe.

.. code:: python

    # Pull all match_log_lists. We will convert each list individually

    def create_match_logs_tables(match_logs_list_urls_x):

        df_30_columns = pd.DataFrame([])
        df_39_columns = pd.DataFrame([])

        count = 0

        for player in match_logs_list_urls_x:
            try: # this may fix "HTTP Error 404: Not Found"
                # urlopen(player)

                new_table = pd.read_html(player)[0]
                new_table.columns = new_table.columns.droplevel()
                new_table['name'] = player.split('/')[-1].replace("-Match-Logs", "")
                
                if new_table.shape[1] == 30:
                    new_table['FBRefID'] = player[(player.find("players/") + len("players/")):(player.find("/matchlogs"))]
                    df_30_columns = df_30_columns.append(new_table, ignore_index=True)
                    count += 1
                    
                    
                if new_table.shape[1] == 39:
                    new_table['FBRefID'] = player[(player.find("players/") + len("players/")):(player.find("/matchlogs"))]
                    df_39_columns = df_39_columns.append(new_table, ignore_index=True)
                    count += 1

                sys.stdout.write("\r{0} percent player urls have just scraped!".format(count / len(match_logs_list_urls_x)*100))
                sys.stdout.flush()

            except:
                pass
        
        return df_30_columns, df_39_columns

    # Creating different length data frames for the first 5000 URLs

    df_30_columns_1, df_39_columns_1 = create_match_logs_tables(match_logs_list_urls[0:5000])

Here the two dataframes generated by the function above are merged into a single dataframe. Only the most relevant columns are stored.

.. code:: python

    #Combining Df_30_columns_1 and df_39_columns_1 to dataframe_1

    cols = ['Date', 'Day', 'Comp', 'Round', 'Venue', 'Result', 'Squad', 'Opponent', 'Start', 'Pos', 'Min', 'Gls', 'Ast', 'PK', 'PKatt', 'Sh', 'SoT', 'CrdY',
        'CrdR', 'Match Report', 'Int', 'name', 'FBRefID']

    df1 = df_39_columns_1
    df2 = df_30_columns_1

    df_final_1 = df1.merge(df2,how='outer', left_on=cols, right_on=cols)

**14. Player Data Dataframe Consolidation.ipynb**

This notebook is used to combine all dataframes produced from the batches above. Here, we also discard unnecessary  columns and clean some NaNs

.. code:: python

    # Concatenating df_final data frames

    df_final_list = [df_final_1, df_final_2, df_final_3, df_final_4, df_final_5, df_final_6, 
                    df_final_7, df_final_8, df_final_9, df_final_10, df_final_11, df_final_12, df_final_13, df_final_14, df_final_15]
    df_final = pd.concat(df_final_list, axis=0, ignore_index=True)

    # Cleaning NaN's from df_final

    df_final.dropna(axis = 0, subset=['Date'], inplace = True)

    # Dropping unwanted columns from df_final

    df_final.drop(columns = ['Match Report'], inplace = True)

**15a. Profile Data Dataframe England.ipynb, 1a.Profile Data Dataframe Italy.ipynb, ...... 15e.Profile Data Dataframe Germany.ipynb**

In these notebooks, we go back to the FBRef website to obtain players' profile information as well as the FBRefIDs, which are unique IDs assigned 
by FBRef to each player. Some relevant profile information such as the birth, height, position and more are considered for the ML models. All 
notebooks follow the same format. Due to the high computational power needed, those 5 notebooks are executed in parallel.

First, we create a function that generates a list of all seasons starting at 2010 from the top 5 leagues. 
Then we apply this function to a one league. In this example, the list will be generated for the English league.

.. code:: python

    def fbref_league_history(league_id = [9,11,12,13,20], first_season = 2010):
        history = []
        for i in league_id:
            comp_history_url = "https://fbref.com/en/comps/" + str(i) + "/history" 
            #print(comp_history_url)

            r=requests.get(comp_history_url)
            soup=BeautifulSoup(r.content, "html.parser")

            find_seasons = soup.find_all(class_ = "left")

            all_seasons_url = []
            for k in range(0, len(find_seasons)):
                if find_seasons[k].get('data-stat') == "season":
                    temp = "https://fbref.com" + find_seasons[k].find_all("a")[0].attrs["href"]
                    all_seasons_url.append(temp)

            history.append(all_seasons_url)
            time.sleep(0.1)

        # All histories in one array
        history  = list(itertools.chain(*history))

        seasons = list(map(lambda x: str(x)+"-"+str(x+1), np.arange(1950, first_season, 1)))
        for i in seasons:
            history = NOTFilter(history, [i])
        del seasons

        return history

    history_england = fbref_league_history(league_id = [9])


This first function generates the list of all teams for all seasons since 2010, and the second function produces the list of all players 
from all of those clubs.

.. code:: python

    def fbref_team_url_history(league_history):
        team_season_url = []
        for league_season_url in league_history:
            r=requests.get(league_season_url)
            soup=BeautifulSoup(r.content, "html.parser")
            teams = soup.find("table").find_all("a")
            teams = list(map(lambda x: "https://fbref.com" + x["href"], teams))
            teams = Filter(teams, ["/en/squads/"])
            team_season_url.append(teams)

        # All histories in one array
        team_season_url  = list(itertools.chain(*team_season_url))
        return team_season_url

    def fbref_team_url_history(league_history):
        team_season_url = []
        for league_season_url in league_history:
            r=requests.get(league_season_url)
            soup=BeautifulSoup(r.content, "html.parser")
            teams = soup.find("table").find_all("a")
            teams = list(map(lambda x: "https://fbref.com" + x["href"], teams))
            teams = Filter(teams, ["/en/squads/"])
            team_season_url.append(teams)

        # All histories in one array
        team_season_url  = list(itertools.chain(*team_season_url))
        return team_season_url

        # Premier League (England) Seasons (England: 9 | Italy: 11 | Spain: 12 | France: 13 | Germany: 20)
        team_season_url_england = fbref_team_url_history(history_england)

An extensive function is created to scrape all players profile information as well as the FBRef ID. Finally, all of the data is exported 
to dataframe called player_data_df_england.csv.

**Important note**

    Refer to the **15a.Profile Data Dataframe England.ipynb** to review the last function. It is not included here since it is very extensive.
    Additionaly, the concatenating of the 5 dataframes is performed in book **17. Consolidate Profile Data Dataframe.ipynb**

.. code:: python

    player_info_england = fbref_player_info(player_url_england)

**16. Extract_Injuries.ipynb**

This notebook is used to scrape players injuries from the years of 2010 to 2021 across the 5 European Leagues. Since we are performing a time series, 
it was decided to only include years from 2010 to 2021. 

Here is where the URLs for every seasons of all leagues are scraped and stored into a list.

.. code:: python

    # Leagues & Seasons
    leagues = [
        "https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1/saison_id/",
        "https://www.transfermarkt.com/bundesliga/startseite/wettbewerb/L1/saison_id/",
        "https://www.transfermarkt.com/laliga/startseite/wettbewerb/ES1/saison_id/",
        "https://www.transfermarkt.com/serie-a/startseite/wettbewerb/IT1/saison_id/",
        "https://www.transfermarkt.com/ligue-1/startseite/wettbewerb/FR1/saison_id/"
    ]

    def all_league_urls(url, season_range = [2010,2021]):
        league_url = []
        for i in url:
            league_url.append(list(map(lambda x : i + str(x), np.arange(season_range[0], season_range[1]+1, 1))))
        league_url  = list(itertools.chain(*league_url))
        return league_url
        
    league_url = all_league_urls(leagues)

Teams URLs are now generated from the list above and stored into a single list 


.. code:: python

    def find_team_urls(league_urls):
        # Teams
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
        team_url = []

        for i in league_urls:
            soup = BeautifulSoup(requests.get(i, headers=headers).content, "html.parser") 
            team_urls = soup.find("table", class_ = "items").find_all("a")
            team_url.append(pd.Series(list(map(lambda x: "https://www.transfermarkt.com" + x["href"], team_urls))).unique().tolist())
        
            # team_urls = soup.find("table", class_ = "items").find_all("a", {"class":"vereinprofil_tooltip"})
            
        team_url  = list(itertools.chain(*team_url))
        links = list(filter(lambda k: 'kader' in k, team_url))
        return links

    team_url = find_team_urls(league_url)

After generating a few more options to obtain the final list of URLs for all desired players, the following function can now pull
the players' injuries

.. code:: python

    def injury_table(url):
        # URL & PLAYER ID
        url = url.replace("profil", "verletzungen")
        pid = url.split("spieler/")[1]

        # Request
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
        r=requests.get(url, headers = headers)
        soup=BeautifulSoup(r.content, "html.parser")
        
        if soup.find("h1") != None:
            name = soup.find("h1").get_text()
            nationality = soup.find("span", {"itemprop":"nationality"}).get_text()
            dateofbirth = soup.find("span", {"itemprop":"birthDate"}).get_text()
            height = soup.find("span", {"itemprop":"height"}).get_text()

        try:
            
            temp = pd.read_html(str(soup.find("table", class_ = "items")))[0]
            
            try:
                # Find page number
                page_numbers = []

                for i in soup.find("div", {'class' : "pager"}).find_all('li'):
                    page_numbers.append(i.find('a')['title'])

                page =len(list(filter(lambda k: 'Page' in k, page_numbers)))
            
                if page > 1:
                    for page_num in np.arange(2, page+1, 1):
                        url2 = url + "/ajax/yw1/page/"+str(page_num)
                        soup2 = BeautifulSoup(requests.get(url2, headers=headers).content, "html.parser")  
                        temp_table2 = pd.read_html(str(soup2.find("table", class_ = "items")))[0]
                        temp = temp.append(temp_table2)
                
            except:
                pass
            
            temp["TMId"]=pid
            temp['name']=name 
            temp['dateofbirth']=dateofbirth
            temp['nationality']=nationality
            temp['height']=height
            
            temp = temp.replace('\n', '', regex=True)
            
            return temp.reset_index(drop=True)
        
        except:
            pass
        
Additionaly, there are other functions that are created to obtain additional attributes that were believed they may contribute to our ML models such 
as 'Retired since:', 'Without Club since:', and more. Last, the final list was run in 3 batches since this step came at high computational cost.

**17. Consolidate Profile Data Dataframe.ipynb**



![hippo](https://media3.giphy.com/media/aUovxH8Vf9qDu/giphy.gif)

player_data_df_england

df_1 = pd.read_csv('player_profile_df_1.csv')
df_2 = pd.read_csv('player_profile_df_2.csv')
df_3 = pd.read_csv('player_profile_df_3.csv')



player_data_df_italy





  Source: www.fbref.com

From FBRef.com we first scraped information from the big 5 European leagues. With that base, we again scraped the website for all the seasons. 
Then we scraped the player information from each of those seasons.  This operation yielded 81,256 player records. Finally we again scraped all 
players' urls to get all the matches that each player had participated in. After going through these 5 iterations of scraping from FBRef.com, we 
obtained a list of 118,283 match logs. With this list we again scraped the website by batches to obtain a final match logs data set, that after 
some NaN cleaning, data type conversion  and dropping unwanted columns, ended up with a DataFrame named consolidated_df that had 3,048,121 rows 
with 47 columns.






player_info_england 
player_info_italy
player_info_spain 
player_info_france 
player_info_germany