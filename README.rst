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

1. FBRED Extrack.ipynb

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

Pull all players' stats for all competitions to end up with a list of all players' URLs for every season they played. Please note that there are more steps during the data scrapping, 
but only the most important ones are shown; refer to the notebooks for the complete code.

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


























