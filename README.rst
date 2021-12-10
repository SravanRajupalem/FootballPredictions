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

    The library needs to be installed:

    $ pip install beautifulsoup4

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
- Jupyter notebooks is the interface used to write, read and produce all scripts for data scrapping, manipulation, visualizations, and creation of all Machine Learning models. 
- Google Drive has been mirrored into our local machines in order to read and write large files through VSCode since our GitHub repository had a limited capacity. 
- An Amazon Web Services(AWS) environment had been generated and linked to VSCode in order to increase computational power and be more productive when building applications 
  that come at a high cost; our local machines experienced multiple memory timeouts and limitations.

Quick Start
~~~~~~~~~~~

1. FBREF Extract.ipynb

In this notebook, we create an extensive list of all match logs for all players and all the seasons they played in. This also includes match logs of other 
competitions such as their previous clubs(even if they played outside of the top 5 leagues) as well as their national team matches. 

Import the following Libraries:

.. code:: python

    import datetime
    from datetime import date
    import requests
    import pprint
    from bs4 import BeautifulSoup
    import pandas as pd
    import re
    import pickle
    from urllib.request import urlopen

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
all players after concatenating all the lists. Thus, a total of 4 batches of 5000 URLs were created to generate the match_logs_urls list.

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

1.5 Append match_url_files.ipynb

In this notebook, we concatenate the match logs lists that were created above to build the final match_log_urls list that contains 
all players' URLs match logs for every single season. This list has 148,478 URLs

.. code:: python

    # Uniting all match logs into a single list:  match_logs_list_urls

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

2. FBREF Player Batch 0-5000.ipynb, 3.FBREF Player Batch 0-5000.ipynB, ........., 13c. FBREF Player Batch 110000-118283 

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

.. code:: python

Here the two dataframes generated by the function above are merged into a single dataframe. Only the most relevant columns are stored.

    #Combining Df_30_columns_1 and df_39_columns_1 to dataframe_1

    cols = ['Date', 'Day', 'Comp', 'Round', 'Venue', 'Result', 'Squad', 'Opponent', 'Start', 'Pos', 'Min', 'Gls', 'Ast', 'PK', 'PKatt', 'Sh', 'SoT', 'CrdY',
        'CrdR', 'Match Report', 'Int', 'name', 'FBRefID']

    df1 = df_39_columns_1
    df2 = df_30_columns_1

    df_final_1 = df1.merge(df2,how='outer', left_on=cols, right_on=cols)


















