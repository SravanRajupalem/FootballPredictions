Football Predictions Capstone Project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: images/stadium.jpg

This capstone project consists of developing an injury predictor that can assist football managers or clubs to make decisions when it comes to investing in football players.

The main features of this project are:

- A high-level API allows data scraping from the FBRef website (https://fbref.com/) to obtain match logs data of signed players from the top 5 European Leagues (Spain, Italy, France, Germani, and England).
- A high-level API allows data scraping from the TransfMarkt website (https://www.transfermarkt.com/) to obtain all possible players' injury data.
- Reference Table of mapped IDs from Fbref players and TransferMarkt sites
- Time series ML models to build injury predictors
- Multiple Visualizations to help the user of all results
- A set of interactive tools to compare players and generate predictions based on players selected by users

**Important note**

    Data scrapping from the FBRef website comes at a high computational cost since each of the top 5 European leagues has about 20 teams, and those teams have an 
    approximate number of 30+ players. The entire data set takes accounts of every single available season.

    The Beautiful Soup Python library was used for pulling data from the web. This requires basic knowledge of how to read and interpret HTML and XML files.

    The following libraries need to be installed:

.. code:: python
    
    !pip install beautifulsoup4
    !pip install pyjsparser
    !pip install js2xml
    !pip install streamlit
    !pip install pycaret
    !pip install scklearn
    

Table of Contents
~~~~~~~~~~~~~~~~~
 - `Overview`_
 - `Scraping the Web for Data`_
 - `Data Manipulation & Feature Engineering`_
 - `Visual Exploration of Data`_
 - `Model Building`_
 - `Blog/Website`_
 - `Citing`_

Overview
~~~~~~~~
- The main programming language used in this project is Python. 
- VSCode is the code-editor employed since it allows the connection of the GitHub repository as well as working cooperatively in real-time.
- Jupyter notebooks is the interface used to write, read and produce all scripts for data scrapping, manipulation, visualizations, and creation of 
  all Machine Learning models. 
- Google Drive has been mirrored into our local machines to read and write large files through VSCode since our GitHub repository had a 
  limited capacity. 
- An Amazon Web Services (AWS) environment had been generated and linked to VSCode in order to increase computational power and be more productive 
  when building applications that come at a high cost; our local machines experienced multiple memory timeouts and limitations.

Scraping the Web for Data
~~~~~~~~~~~~~~~~~~~~~~~~~

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

    # Grafikten Data ??ekmek i??in
    import re
    import js2xml
    from itertools import repeat    
    from pprint import pprint as pp

    # Configurations
    import warnings
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pd.set_option('display.max_columns', None)
    
    # Machine Learning Models
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score,   precision_recall_curve
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold, StratifiedKFold
    from sklearn.datasets import make_hastie_10_2
    from sklearn.ensemble import GradientBoostingClassifier
    from imblearn.pipeline import make_pipeline as make_pipeline_with_sampler
    from imblearn.under_sampling import RandomUnderSampler
    from pycaret.classification import * 

**1. FBREF Extract.ipynb**

.. image:: images/top5.png

In this notebook, we create an extensive list of all Big 5 European leagues match logs for all players and all the seasons they played from the FBRef website. 
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

Here we pull all players' stats for all competitions to conclude with a list of all players' URLs for every season they played. Please note that there are more 
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

The following function had to be applied in multiple batches since this operation required high computation.

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

Once this function is created, we imported the mapping table of FBRefIDs and TMIDs to only pull data from the intersection of FBRefIDs and TMIDs. This step allowed us to avoid an unnecessary effort to pull match logs for players that we will not use.

.. code:: python    

    fbref_to_tm_mapping = pd.read_csv('.../CSV files/fbref_to_tm_mapping.csv', encoding='latin-1')
    player_all_competitions_filtered = player_all_competitions_df.merge(fbref_to_tm_mapping, left_on='FBRefID', right_on='FBRefID', how='inner')
    player_all_competitions_filtered_list = list(player_all_competitions_filtered[0])


Here we were able to generate a list of 51,196 URLs for a total of 5,192 players. This list of URLs is used to 
scrape all match logs URLs of all the consolidated players. The list called **match_logs_list** at first,
but then we exported as csv named **match_logs_list_urls.csv**.

.. code:: python   

    # Total length of player_all_competitions is 5192
    
    match_logs_list = []

    count = 0
    for i in range(len(player_all_competitions_filtered_list)):
        match_logs_list.extend(get_player_match_logs(player_all_competitions_filtered_list, i))
        count += 1
        sys.stdout.write("\r{0} percent".format((count / len(player_all_competitions_filtered_list)*100)))
        sys.stdout.flush()

**2a. FBREF Player Batch 0-5000.ipynb** 
**2b. FBREF Player Batch 5000-10000.ipynb**
**2c. FBREF Player Batch 10000-15000.ipynb**
**2d. FBREF Player Batch 15000-20000.ipynb**
**23. FBREF Player Batch 20000-25000.ipynb**
**2f. FBREF Player Batch 25000-30000.ipynb**
**2g. FBREF Player Batch 30000-40000.ipynb**
**2h. FBREF Player Batch 40000-51196.ipynb** 

It is time to perform the real data scrapping. Here, we are pulling data from the created list, which contains a total of 51,196 URLs. 
When executing the function below, we are extracting the match logs of all seasons for every single player. In addition, we found that some players 
have match logs that contain 30 attributes or columns while other players have match logs with 39 attributes. Thus, players' match logs are 
appended to two dataframes of 30 columns and 39 columns, respectively. 

**Important note**

    This step took a significant amount of memory usage. Therefore, it was necessary to run the **match_logs_list_urls.csv** in multiple batches. 
    A total of 8 notebooks were created in order to run all batches in parallel. The function below is used across all FBREF Player Batch notebooks; 
    this is an example of the first batch. In the end, all dataframes are concatenated together to produce a single dataframe.


.. code:: python    

    # Pull all match_log_lists_x tables. We will convert each list individually WORK IN PROCESS

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
    
    # Creating different length data frames - Here is where we update the URLs that we will use

    df_30_columns_1, df_39_columns_1 = create_match_logs_tables(match_logs_list_urls[0:5000])    
    
    #Combining Df_30_columns_1 and df_39_columns_1 to dataframe_1

    cols = ['Date', 'Day', 'Comp', 'Round', 'Venue', 'Result', 'Squad', 'Opponent', 'Start', 'Pos', 'Min', 'Gls', 'Ast', 'PK', 'PKatt', 'Sh', 'SoT', 'CrdY',
           'CrdR', 'Match Report', 'Int', 'name', 'FBRefID']

    df1 = df_39_columns_1
    df2 = df_30_columns_1

    df_final_1 = df1.merge(df2,how='outer', left_on=cols, right_on=cols)

    print(df1.shape)
    print(df2.shape)

At the end, we excuted all remaining notebooks and exported them as csv files with the goal of concatenating them into a single dataframe. 
We do this in the next notebook.

**3. Player Data Dataframe Consolidation.ipynb**

This notebook is used to combine all dataframes produced from the batches above. Here, we also discard unnecessary columns and clean some NaNs

.. code:: python

    # Concatenating df_final data frames

    df_final_list = [df_final_1, df_final_2, df_final_3, df_final_4, df_final_5, df_final_6, 
                    df_final_7, df_final_8, df_final_9, df_final_10, df_final_11, df_final_12, df_final_13, df_final_14, df_final_15]
    df_final = pd.concat(df_final_list, axis=0, ignore_index=True)

    # Cleaning NaN's from df_final

    df_final.dropna(axis = 0, subset=['Date'], inplace = True)

    # Dropping unwanted columns from df_final

    df_final.drop(columns = ['Match Report'], inplace = True)
    
    # Converting date columns to datetime

    consolidated_df_final['Date'] = pd.to_datetime(consolidated_df_final['Date'])

**4a. Profile Data Dataframe England.ipynb**
**4b. Profile Data Dataframe Italy.ipynb**
**4c. Profile Data Dataframe Spain.ipynb**
**4d. Profile Data Dataframe France.ipynb**
**4e. Profile Data Dataframe Germany.ipynb**

In these notebooks, we go back to the FBRef website to obtain players' profile information as well as the FBRefIDs, which are unique IDs assigned 
by FBRef to each player. Some relevant profile information such as birth date, height, position, and more are considered for the ML models. All 
notebooks follow the same format. Due to the high computational power needed, those 5 notebooks are executed in parallel.

First, we create a function that generates a list of all seasons starting in 2010 from the top 5 leagues. 
Then we apply this function to one league. In this example, the list will be generated for the English league.

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

An extensive function is created to scrape all players' profile information as well as the FBRef ID. Finally, all of the data is exported 
to dataframe called **player_data_df_england.csv**.

**Important note**

    Refer to the **15a.Profile Data Dataframe England.ipynb** to review the last function. It is not included here since it is very extensive.
    Additionally, the concatenating of the 5 dataframes is performed in book **17. Consolidate Profile Data Dataframe.ipynb**

.. code:: python

    player_info_england = fbref_player_info(player_url_england)

**5. Extract_Injuries.ipynb**

.. image:: images/zidane.gif

This notebook is used to scrape players injuries from the years 2010 to 2021 across the 5 European Leagues, and obtain additional players'
profile data from the TransferMarkt site. Since we are performing a time series, it was decided to only include years from 2010 to 2021. 

Here is where the URLs for every season of all leagues are scraped and stored into a list.

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

After generating a few more steps to obtain the final list of URLs for all desired players, the next 2 following functions can now pull
the players' injuries. Then, this is exported into a dataframe called **'player_injuries_df.csv'**.

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
    
    player_urls = list(tm_player_url_df['TMURL'])

    player_urls =list(filter(lambda k: 'profil' in k, player_urls))

    player_injuries_df = pd.DataFrame(columns=['Season', 'Injury', 'from', 'until', 'Days', 'Games missed', 'TMId', 'name'])

    for i in player_urls:
        df = injury_table(i)
        player_injuries_df = player_injuries_df.append(df)
        sys.stdout.write("\r{0} player injuries have just scraped from TM!".format(len(player_injuries_df)))
        sys.stdout.flush()

    player_injuries_df.to_csv('player_injuries_df.csv', index=False)  
        
Further, tother functions are created to obtain a new dataframe that captures profile data with additional attributes that 
contribute to our ML models such as 'Retired since:', 'Without Club since:', and more. Last, this final dataframe is generated in 3 batches 
since, again, the data scraping comes at a high computational cost. These files are exported to 3 dataframes player_profile_df_1.csv,
player_profile_df_2.csv, and player_profile_df_3.csv.

Data Manipulation & Feature Engineering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**6. Consolidate Profile Data Dataframe.ipynb**

This is the most extensive notebook in our entire repository. Here is where we combine all created dataframes to build the main dataframe. Thus, be prepared
to spend some time reading this notebook. 

.. image:: images/guardiola.gif

First, we begin by importing all CSV files that have been previously generated, including some that were generated in batches. Then we merged those 
together.

Here are all the CSV files that are called:

.. code:: python

    # Player profile from FBRef site - 5 dataframes are concatenated into a single dataframe - shape is (35827, 15)
    player_info_england = pd.read_csv('.../Dataframes/player_data_df_england.csv')
    player_info_italy = pd.read_csv('.../Dataframes/player_data_df_italy.csv')
    player_info_spain = pd.read_csv('.../Dataframes/player_data_df_spain.csv')
    player_info_france = pd.read_csv('.../Dataframes/player_data_df_france.csv')
    player_info_germany = pd.read_csv('.../Dataframes/player_data_df_germany.csv')

    player_inf_lst = [player_info_england, player_info_italy, player_info_spain, player_info_france, player_info_germany]
    player_info_df = pd.concat(player_inf_lst)

    # Cleaning repeated players - shape is (10720, 15)
    player_info_df_nodups = player_info_df.drop_duplicates()

    # Player profiles from TransferMarkt - 3 dataframes are concatenated into a single dataframe - shape is (12902, 41)
    df_1 = pd.read_csv('.../player_profile_df_1.csv')
    df_2 = pd.read_csv('.../player_profile_df_2.csv')
    df_3 = pd.read_csv('.../player_profile_df_3.csv')

    tm_profile_df = pd.concat([df_1, df_2])
    tm_profile_df = pd.concat([tm_profile_df, df_3])

    # Player injuries from TransferMarkt - length is 55216
    player_injuries_df = pd.read_csv('.../Dataframes/player_injuries_df.csv')

    # Reference table - this is used to map FBRef IDs (FBRefID) to TransferMarkt IDs (TMID)
    fbref_to_tm_df = pd.read_csv('.../CSV files/fbref_to_tm_mapping.csv')

    # Pull the IDs from the URLs
    fbref_to_tm_df['FBRefID'] = fbref_to_tm_df['UrlFBref'].str.split('/').str[5]
    fbref_to_tm_df['TMID'] = fbref_to_tm_df['UrlTmarkt'].str.split('/').str[6]

    # Merging on intersection of player_injuries_df and fbref_to_tm_df on columns TMId and TMID respectively - shape is (32660, 14)
    player_injuries_df_2 = pd.merge(left=player_injuries_df, right=fbref_to_tm_df, left_on='TMId', right_on='TMID', how='inner')

    # Merging Player Injuries with FBRef Profiles
    player_injuries_info_df = pd.merge(left=player_injuries_df_2, right=player_info_df, left_on='FBRefID', right_on='FBRefId', how='inner')

    # Merge with TM Profile information
    player_injuries_profile_final = pd.merge(left=player_injuries_info_df, right=tm_profile_df, left_on='TMId', right_on='TMId', how='inner')

This is just the beginning...

.. image:: images/referee.gif

There is a great number of steps taken on this notebook, we only highlight the ones we believe are the most relevant. Steps like
removing duplicates, dropping NaNs, updating the column types, and any other basic operations are excluded. We also do some testing in order
to understand what data cleaning is required and more. Please refer to the **6. Consolidate Profile Data Dataframe.ipynb** for the 
complete notebook.

Here we create some important features that are considered for our time series models.

.. code:: python

    # Creating new columns of the week and year a player gets injured as well as the week the player is released

    player_injuries_profile_final = player_injuries_profile_final[player_injuries_profile_final['from'] != '-']
    player_injuries_profile_final = player_injuries_profile_final[player_injuries_profile_final['until'] != '-']
    player_injuries_profile_final['injury_year'] = player_injuries_profile_final['from'].apply(lambda x: datetime.strptime(x, '%b %d, %Y').year)
    player_injuries_profile_final['injury_week'] = player_injuries_profile_final['from'].apply(lambda x: datetime.strptime(x, '%b %d, %Y').strftime('%V'))
    player_injuries_profile_final['release_week'] = player_injuries_profile_final['until'].apply(lambda x: datetime.strptime(x, '%b %d, %Y').strftime('%V'))
    player_injuries_profile_final['from'] = pd.to_datetime(player_injuries_profile_final['from'])
    player_injuries_profile_final['until'] = pd.to_datetime(player_injuries_profile_final['until'])

    # Creating new columns - player's team wins, loses or draws a game, also add a column to highlight when player starts playing
    # since the beginning of the match
    total_match_logs_df.loc[total_match_logs_df['Result'].str[0] == 'W', 'Won'] = 1
    total_match_logs_df.loc[total_match_logs_df['Result'].str[0] != 'W', 'Won'] = 0

    total_match_logs_df.loc[total_match_logs_df['Result'].str[0] == 'L', 'Loss'] = 1
    total_match_logs_df.loc[total_match_logs_df['Result'].str[0] != 'L', 'Loss'] = 0

    total_match_logs_df.loc[total_match_logs_df['Result'].str[0] == 'D', 'Draw'] = 1
    total_match_logs_df.loc[total_match_logs_df['Result'].str[0] != 'D', 'Draw'] = 0

    total_match_logs_df.loc[total_match_logs_df['Start'] == 'Y', 'Games_Start'] = 1
    total_match_logs_df.loc[total_match_logs_df['Start'] != 'Y', 'Games_Start'] = 0

This is a critical step. Here we aggregate all columns at the week level. Our final dataset will contain all players' profile data,
match logs, and injuries at the week level. For example, a football player plays 2 entire games within a week; then the player is playing 
a total of 180 minutes. The same applies when a player scores in multiple games. This step aggregates all column values with the groupby 
function and the sum() operator. Also, we can now merge the player_injuries_profile_final. 

.. code:: python

    # Grouping total_match_logs_df_2 by name, FBRefID, week and year    
    total_match_logs_df_3 = total_match_logs_df_2.groupby(by=['name', 'FBRefID','week', 'year', 'Date']).sum().reset_index()

    # Merging total_match_logs_df with player_injuries_profile_final
    complete_final_df = pd.merge(left=total_match_logs_df_3, right=player_injuries_profile_final, left_on=['week', 'year', 'Date', 'FBRefID'], right_on=['current_week', 'current_year', 'current_date', 'FBRefID'], how='outer')

Now that this dataframe is at the week level, we proceed to develop more columns

.. code:: python

    # Creating variable 'was_match' to know which rows are matches (real games) and which rows are not
    complete_final_df.loc[complete_final_df['Min'].isnull(), 'was_match'] = 0
    complete_final_df.loc[complete_final_df['Min'].isnull() == False, 'was_match'] = 1

This is another critical step for our time series models. Here we add the weeks when players did not play and fill those with 0s. 
In other words, if a player didn't play a certain week, we add a row and populate all the date columns accordingly and the remaining 
columns are filled with 0s. In addition, we perform another merge so we can only filter on players from FBRef.

.. code:: python

    def get_player_dates(fb_ref_id_list, df):
        new_player_df = pd.DataFrame([])
        
        count = 0
        
        for fbref in fb_ref_id_list:
            player_df = df[df['FBRefID'] == fbref].copy()
            range = pd.date_range(start=player_df['date'].min(), end=player_df['date'].max(), freq='W')
            range_df = pd.DataFrame(range).reset_index()
            range_df['date'] = range_df[0]
            range_df.drop(columns={0, 'index'}, inplace=True)
            range_df['date'] = pd.to_datetime(range_df['date']) #.apply(lambda x: x.strftime("%Y-%m-%d"))
            player_df['date'] = pd.to_datetime(player_df['date'])
            player_merge = player_df.merge(range_df, left_on='date', right_on='date', how='outer').sort_values('date')
            player_merge['FBRefID'] = player_merge['FBRefID'].ffill()
            
            if new_player_df.shape == (0, 0):
                new_player_df = player_merge.sort_values(['FBRefID', 'date'])        
            else:
                new_player_df = new_player_df.append(player_merge.sort_values(['FBRefID', 'date']), ignore_index=True)
            
            count += 1
            sys.stdout.write("\r{0} percent FBRefID's have been processed!".format(count / len(fb_ref_id_list)*100))
            sys.stdout.flush()

        new_player_df['agg_week'] = new_player_df['agg_week'].fillna(new_player_df['date'].dt.isocalendar().week)
        new_player_df['agg_year'] = new_player_df['agg_year'].fillna(new_player_df['date'].dt.year)
        
        return new_player_df

    new_player_df = get_player_dates(unique_FBRefIDs, complete_final_df_4)

The following columns are added as dummy variables. Once we were able to complete the final merge, these columns were considering that 
these features could be of great importance to improve our models.

.. code:: python

    # Assigning Dummy Variables for player position from 'Position:'
    new_player_df.loc[new_player_df['Position:'].isnull(), 'Position:'] = ''

    new_player_df['defender'] = np.where(new_player_df['Position:'].str.contains('Defender'), 1, 0)
    new_player_df['attacker'] = np.where(new_player_df['Position:'].str.contains('attack'), 1, 0)
    new_player_df['midfielder'] = np.where(new_player_df['Position:'].str.contains('midfield'), 1, 0)
    new_player_df['goalkeeper'] = np.where(new_player_df['Position:'].str.contains('Goalkeeper'), 1, 0)

    new_player_df['right_foot'] = np.where(new_player_df['Foot'].str.contains('RIGHT'), 1, 0)
    new_player_df['left_foot'] = np.where(new_player_df['Foot'].str.contains('LEFT'), 1, 0)

    new_player_df['Injury'] = new_player_df['Injury'].astype(str)
    new_player_df.loc[new_player_df['Injury'] == '0', 'injury_count'] = 0
    new_player_df.loc[new_player_df['Injury'] != '0', 'injury_count'] = 1 

    new_player_df['cum_injury'] = new_player_df.groupby(['FBRefID'])['injury_count'].cumsum()

    new_player_df['age'] = round((pd.to_datetime(new_player_df['date']) - pd.to_datetime(new_player_df['Birth'])) / timedelta(days=365), 0)

Also, we believed competitions or tournaments where players participated could influence our model, especially when players are on international duty during major tournaments such as the world qualifiers. Thus, we created dummy variables to identify what tournament players played and added those as new features.

.. code:: python

    def make_dummies(df, feature, suffix):
        feature_list = list(df[feature].unique())

        for col in feature_list:
            df[col] = 0

        for row in range(len(df)):
            for features in feature_list:
                 if df[feature].iloc[row] == features:
                     df[features + suffix].iloc[row] = 1

        return df
   
    feature_list = ['Serie A', 'Premier League', 'La Liga', 'Ligue 1', 'Bundesliga', 'Champions Lg', 'Europa Lg', 'FIFA World Cup', 'UEFA Nations League', 'UEFA Euro', 'Copa      Am??rica']

    for col in feature_list:
        total_match_logs_df.loc[total_match_logs_df['Comp'] == col, col] = 1
        total_match_logs_df.loc[total_match_logs_df['Comp'] != col, col] = 0
  
Other important features are the injury count as well the previous injury weeks, and the weeks that players got injured. Here is how we did it:

.. code:: python

    new_player_df.loc[(new_player_df['injury_count']== 1) & (new_player_df['injury_count'].shift(1) == 1), 'unique_injury_count'] = 0
    new_player_df.loc[(new_player_df['injury_count']== 1) & (new_player_df['injury_count'].shift(1) == 0), 'unique_injury_count'] = 1
    new_player_df.loc[(new_player_df['injury_count']== 0) & (new_player_df['injury_count'].shift(1) == 0), 'unique_injury_count'] = 0
    new_player_df.loc[(new_player_df['injury_count']== 0) & (new_player_df['injury_count'].shift(1) == 1), 'unique_injury_count'] = 0

    new_player_df['cum_injury_total'] = new_player_df.groupby(['FBRefID'])['unique_injury_count'].cumsum()

    new_player_df["previous_injury_week"] = new_player_df.groupby(["FBRefID", "cum_injury_total"])["cum_week"].transform("first")

    new_player_df.loc[new_player_df['previous_injury_week'] == 0, 'weeks_since_last_injury'] = 0
    new_player_df.loc[new_player_df['previous_injury_week'] != 0, 'weeks_since_last_injury'] = new_player_df["cum_week"] - new_player_df["previous_injury_week"]

We also added more other features before and after we created this notebook.
In the end, we ended with a final dataset of shape (1910255, 169).

Are we done?

.. image:: images/cristiano.gif

..... for now .....

**xx. Preparing Features for Models.ipynb**
Although some features had already been created for our models as we have been consolidating our final dataset, there were still some
features that we were reengineering as we build our models. Sometimes, adding basic columns such the as next example could help our 
model to learn better and provide more accurate predictions. Additionally, it is worth mentioning that there were a number of features
that we created in this notebook, but were later removed since they didn't add value to our models. We have not included those. 

This new feature assigns a 1 when a player is injured, otherwise a 0 is assigned.

.. code:: python

    # Creating 'injured' column

    dataset.loc[dataset['Injury'] != '0', 'injured'] = 1
    dataset.loc[dataset['Injury'] == '0', 'injured'] = 0

    # Creating target column 'injured_in_one_week' and creating cumulative features
    
    def shift_by_time_period(df, shift_factor, column):
        df[column + '_in_' + str(shift_factor) + '_week'] = df.groupby('FBRefID')[column].shift(shift_factor*-1)
        return df

    dataset = shift_by_time_period(dataset, 1, 'injured')
    dataset = shift_by_time_period(dataset, 4, 'injured')
    dataset = shift_by_time_period(dataset, 12, 'injured')
    dataset = shift_by_time_period(dataset, 26, 'injured')
    dataset = shift_by_time_period(dataset, 52, 'injured')

    dataset = shift_by_time_period(dataset, 1, 'injury_count')
    dataset = shift_by_time_period(dataset, 4, 'injury_count')
    dataset = shift_by_time_period(dataset, 12, 'injury_count')
    dataset = shift_by_time_period(dataset, 26, 'injury_count')
    dataset = shift_by_time_period(dataset, 52, 'injury_count')

    dataset = shift_by_time_period(dataset, 1, 'cum_injury')
    dataset = shift_by_time_period(dataset, 4, 'cum_injury')
    dataset = shift_by_time_period(dataset, 12, 'cum_injury')
    dataset = shift_by_time_period(dataset, 26, 'cum_injury')
    dataset = shift_by_time_period(dataset, 52, 'cum_injury')
    
The following features are used to create a 'cum_sum' column which will serve as base for cummulative features that will be used for our models. 
We do this by applying the groupby function and the cumsum() operator.

.. code:: python

    dataset['cum_sum'] = dataset['injured'].cumsum()
    
    # Creating function to add cummulative columns

    def cummulative_sum(dataset, cum_column, original_column):
        dataset[cum_column] = dataset.groupby(['FBRefID', 'cum_sum'])[original_column].cumsum()
        return dataset
            
    # Creating cummulative variables
    cum_cols = ['Min', 'Gls', 'Ast', 'PK', 'PKatt', 'Sh', 'SoT', 'CrdY', 'CrdR', 'Touches', 'Press', 'Tkl', 'Int', 'Blocks', 'xG', 'npxG', 'xA', 
        'SCA', 'GCA', 'Cmp', 'Att', 'Prog', 'Carries', 'Prog.1', 'Succ', 'Att.1', 'Fls', 'Fld', 'Off', 'Crs', 'TklW', 'OG', 'PKwon', 'PKcon', 'Won', 
        'Loss', 'Draw', 'was_match']

    for var in cum_cols:
        cummulative_sum(dataset, var+'_cum', var)


Visual Exploration of Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

        

Model Building
~~~~~~~~~~~~~~

Now it is time to start buiding our machine learning models that will be used to generate predcitions. At first, we attempted to use 
scklearn which provided us results, but we decided to use pycaret since it run faster when we use it as a tool inside our blog. Here, we will
just show you a few samples of what we did using pycaret.

Please refer to the the following notebooks:

**8. Modelling Without Pycaret.ipynb**
**8a. One Week Prediction.ipynb**
**8b. One Month Prediction.ipynb**
**8c. One Quarter Prediction.ipynb**
**8d. One Semester Prediction.ipynb**
**8e. One Year Prediction.ipynb**
**8f. Interactive Prediction Tool Development.ipynb**

First, we created a variable called "injured_pred", this has been updated throughout all notebooks so we could update the time range for our predictions.
Additionally, we created to list of simple features and a second list of extended features as we were doing some trial and error using multiple features
for our models.

Here we will only show what we did for the **8a. One Week Prediction.ipynb** notebook

.. code:: python

    injured_pred = 'injured_in_1_week'

    simple_features = ['Height', 'Weight', 'age','cum_injury_total', 'weeks_since_last_injury', 'Min_cum','Serie A_cum',
    'Premier League_cum', 'La Liga_cum', 'Ligue 1_cum', 'Bundesliga_cum', 'Champions Lg_cum', 'Europa Lg_cum', 'FIFA World Cup_cum', 'UEFA Nations League_cum', 'UEFA Euro_cum',
    'Copa Am??rica_cum', 'Away_cum', 'Home_cum', 'Neutral_cum']

    extended_features = ['Height', 'Weight', 'defender', 'attacker', 'midfielder', 'goalkeeper', 'right_foot', 'age', 'cum_injury_total', 'weeks_since_last_injury', 'Min_cum', 'Gls_cum', 'Ast_cum', 'PK_cum', 'PKatt_cum',
    'Sh_cum', 'SoT_cum', 'CrdY_cum', 'CrdR_cum', 'Touches_cum', 'Press_cum', 'Tkl_cum', 'Int_cum', 'Blocks_cum', 'xG_cum', 'npxG_cum', 'xA_cum', 'SCA_cum', 'GCA_cum', 'Cmp_cum',
    'Att_cum', 'Prog_cum', 'Carries_cum', 'Prog.1_cum', 'Succ_cum', 'Att.1_cum', 'Fls_cum', 'Fld_cum', 'Off_cum', 'Crs_cum', 'TklW_cum', 'OG_cum', 'PKwon_cum','PKcon_cum', 'Serie A_cum',
    'Premier League_cum', 'La Liga_cum', 'Ligue 1_cum', 'Bundesliga_cum', 'Champions Lg_cum', 'Europa Lg_cum', 'FIFA World Cup_cum', 'UEFA Nations League_cum', 'UEFA Euro_cum',
    'Copa Am??rica_cum', 'Away_cum', 'Home_cum', 'Neutral_cum']

We split out dataset to create a train set and test set for our models. This analysis requries a different approach as it is a time series 
analysis for each player. Hence, the number of weeks each player plays was determined and the first 75% of the player's career in weeks was 
allocated to the training dataset and the remaining 25% was allocated to the test set.

.. code:: python

    # Get Train Test Split
    df_train = dataset[dataset['cum_week'] <= dataset["train_split"]].dropna()
    df_test = dataset[dataset['cum_week'] > dataset["train_split"]].dropna()

    X_train = df_train[extended_features]
    y_train = df_train[injured_pred]

    X_test = df_test[extended_features]
    y_test = df_test[injured_pred]


The following operation allows to compare multiple models at once. This method is very powerfull since we avoid the need to create models
individually. We run a range of classification algorithms all with one simple line of code. This produces an output with accuracy, AUC, 
recall, precision and F1 across all the models for easy comparison.

.. image:: images/image26.PNG

.. code:: python

    # df = pull()
    # df.to_csv('results_1_year.csv', index=False)
    df = pull().sort_values(by=['F1'], ascending=False)


    # Select best model
    model = create_model(df.index[0], fold=10)
    save_model(model, 'model_1_week')

.. image:: images/image27.PNG

.. code:: python

    tuned_model = tune_model(model, optimize = 'F1')

.. image:: images/image28.PNG


Blog/Website
~~~~~~~~~~~~

Streamlit is an open-source Python library that makes it easy to create and share custom web apps for machine learning and data science. We used Streamlit to
create a blog where we could share our ideas for this project and also offer an interactive tool that allows you to compare players at multiple levels, generate visualizations, and more. You can now review our blog and start playing with our custom apps.

First, we needed to do some research to understand how to use Streamlit and to decide if we wanted to use it. It turned out that this library was manageable to learn as compared to others we tested. 

You need the following installion to make Streamlit is available.

.. code:: python

    pip install streamlit

Although we won't go into much details, we want to share some samples of the custom app we developed with the help of Streamlit.

Here we can create a selection box where we created multiple sections for our blogs. Here, users are able to select a section of our website.

.. code:: python

    section = st.sidebar.selectbox("Sections", ("Introduction", "Scraping the Web for Data", "Data Manipulation & Feature Engineering", 
        "Visual Exploration of Data", "Model Building", "Injury Prediction", "Interactive Exploration Tool (BETA)", 
        "Interactive Injury Prediction Tool (BETA)", "Conclusions and Future Work"))

By using st.write(), we added complete sentences and paragraphs in our blog. Along with those, we incorporated images to make our
blog more entertaining and to keep the users engaged. We first loaded the images to our GitHub repository, and then called the images and display those
using the following:

.. code:: python

    st.write("The first major decision was that we would only get information from the five most competitive soccer leagues in \
        the world: Premier League (England), La Liga (Spain), Bundesliga (Germany), Ligue 1 (France) and the Serie A (Italy). \
        The reason for this decision was that we thought that these leagues would have better player documentation.")
    
    img4 = Image.open("images/image4.png")
    st.image(img4)
    
Here is the result:

.. image:: images/image23.PNG

Although we are not going to every single step of how we designed all interactive apps, here is the complete coding for the "Compare Player's Injury History".
As you can notice, we are using the altair library to build the visualizations. Go to our blog to interact with this tool.

.. code:: python

    elif section == "Interactive Exploration Tool (BETA)":
        st.header('Interactive Exploration Tool (BETA)')
        @st.cache  # ???? Added this
        def get_df():
            path = 'dataframes_blog/dataset_for_model_final.parquet'
            return pd.read_parquet(path)

        dataset = copy.deepcopy(get_df())

        # Plotting Chart 1: Compare Players' Injury History

        st.subheader("Compare Players' Injury History")

        sorted_unique_player = dataset['name'].sort_values().unique()
        player1 = st.selectbox('Player 1 Name (type or choose):',sorted_unique_player)
        player2 = st.selectbox('Player 2 Name (type or choose):',sorted_unique_player)
        player3 = st.selectbox('Player 3 Name (type or choose):',sorted_unique_player)

        @st.cache(allow_output_mutation=True)
        def chart1(player1, player2, player3): 
            df1_1 = dataset[dataset['name'] == player1][['cum_week', 'name', 'cum_injury_total']]
            df1_2 = dataset[dataset['name'] == player2][['cum_week', 'name', 'cum_injury_total']]
            df1_3 = dataset[dataset['name'] == player3][['cum_week', 'name', 'cum_injury_total']]

            df = pd.concat([df1_1, df1_2, df1_3])

            chart1 = alt.Chart(df).mark_line().encode(x=alt.X('cum_week:Q', axis=alt.Axis(labelAngle=0)), y='cum_injury_total:Q', color='name'). \
                properties(width=800, height=300)

            return chart1

        chart1_output = copy.deepcopy(chart1(player1, player2, player3))
        st.altair_chart(chart1_output, use_container_width=False)
        
        
Citing 
~~~~~~

- FBref.com. 2021. Football Statistics and History | FBref.com. [online] Available at: <https://fbref.com/en/>
- Transfermarkt.com. 2021. Football transfers, rumours, market values and news. [online] Available at: <https://www.transfermarkt.com/>
- Medium. 2021. Beating soccer odds using Machine Learning?????????Project Walkthrough. [online] Available at: <https://medium.com/analytics-vidhya/beating-soccer-odds-using-machine-learning-project-walkthrough-a1c3445b285a>
- Kaggle.com. 2021. Comprehensive data exploration with Python. [online] Available at: <https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python>
- Medium. 2021. Creating Multipage applications using Streamlit (efficiently!). [online] Available at: <https://towardsdatascience.com/creating-multipage-applications-using-streamlit-efficiently-b58a58134030>




