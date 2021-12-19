import streamlit as st
from PIL import Image
import pandas as pd
import altair as alt
from pathlib import Path
import os
import requests
import copy
import dask.dataframe as dd
import streamlit.components.v1 as components

imglogo = Image.open("images/logo.png")

section = st.sidebar.selectbox("Sections", ("Introduction", "Scraping the Web for Data", "Data Manipulation & Feature Engineering", 
    "Visual Exploration of Data", "Model Building", "Injury Prediction", "Interactive Exploration Tool (BETA)", 
    "Interactive Injury Prediction Tool (BETA)", "Conclusions, Challenges, and Future Work"))

if section == "Introduction":
    imgstadium = Image.open("images/stadium1.png")
    st.image(imgstadium, width=700)

    st.title("Sooner or later?  Walkthrough to predict when an elite soccer player will get injured.")

    st.write("Sravan Rajupalem (sravanr@umich.edu)") 
    st.write("Renzo Maldonado (renzom@umich.edu)")
    st.write("Victor Ruiz (dsvictor@umich.edu)")
    st.markdown("***")


    st.write("""For quite a while, 'Sports Analytics' has been the buzz-word in the world of Data Science. Magically using complex 
        algorithms, machine learning models and neural networks to predict sports results and players' performance attract the interest 
        of people for different reasons. Soccer is probably one of the most unpredictable sports out there. In the hope of aiding soccer 
        managers' decisions, we decided to apply Data Science tools to predict how likely a player was to have an injury within a 
        certain time frame.""")

    st.image(imglogo, width=250)

    st.write("""Presenting Providemus, a tool to predict when a player will get injured.  By using data from the most reliable 
        international soccer sources, our data scientists have been able to train machine learning models to predict with
        considerable accuracy when will a player get injured. The time frame the tool has are if a player will get injured
        during the next week, month, quarter, semester or year. This system is meant to be used as a complementary tool for 
        soccer managers in their decisions to play or rest their players.""")

elif section == "Scraping the Web for Data":
    imglogo = Image.open("images/logo.png")
    st.image(imglogo, width=250)

    imgcorner = Image.open("images/corner.jpg")
    st.image(imgcorner, width=700)
    
    st.header('Scraping the Web for Data')
    st.write("<p style='text-align: justify; font-size: 15px'>We hunted the web to get the most information we could about soccer \
        players and matches. After scanning several options our runners up due to the completeness of their data \
        were:  fbref.com and transfermarkt.com.</h1>", unsafe_allow_html=True)
    img = Image.open("images/image1.png")
    img2 = Image.open("images/image2.png")
    img3 = Image.open("images/image3.png")
    st.image(img)
    st.image(img2)
    st.image(img3)
    st.write("<p style='text-align: justify; font-size: 15px'>The first major decision was that we would only get information from the five most competitive soccer leagues in \
        the world: Premier League (England), La Liga (Spain), Bundesliga (Germany), Ligue 1 (France) and the Serie A (Italy). \
        The reason for this decision was that we thought that these leagues would have better player documentation.</h1>", unsafe_allow_html=True)
    img4 = Image.open("images/image4.png")
    st.image(img4)
    st.write("<p style='text-align: justify; font-size: 15px'>From FBRef we first scraped urls from the big 5 European leagues. With that base, we again scraped the website \
        for all the seasons for each league. Then we scraped the players' urls from each of all available seasons of the top 5. \
        This operation yielded a list of 78,959 unique records. Those embedded urls contained an identifier (FBRefID) for each of the \
        19,572 players from this fbref.com. Moreover, since we intended to scrape complementary players' data from the TransferMarkt \
        website, we decided to only pull data for the players whose information was available on both sites.</h1>", unsafe_allow_html=True)   
    st.write("<p style='text-align: justify; font-size: 15px'>Next, we had to use a handy mapping dataset called fbref_to_tm_mapping that links the websites' unique identifiers \
        FBRefID and TMID (TransferMarkt ID), which we downloaded from [Jason Zickovic](https://github.com/JaseZiv/worldfootballR_data/tree/master/raw-data/fbref-tm-player-mapping/output) \
        (kudos!). Via string conversions and splitting we extracted the FBRefID's from the generated list and decided to only scrape \
        the match logs for those players from the FBRef site.</h1>", unsafe_allow_html=True)
    img5 = Image.open("images/image5.png")
    st.image(img5)
    st.write("<p style='text-align: justify; font-size: 15px'>This effort helped us reduce a significant amount of memory usage when performing the data scrapping given that only 5,192 \
        players had attainable data from both sites. Now we can execute another pull, but this time we obtained a list of 51,196 the complete \
        match logs urls of all the consolidated players.</h1>", unsafe_allow_html=True)
    img5a = Image.open("images/image5a.PNG")
    st.image(img5a)
    st.write("<p style='text-align: justify; font-size: 15px'>This is where the real data scraping of the players' match logs begun. The extraction of all players matches required high \
        computation; thus, our team divided the data extraction in multiple batches, where we extracted the match logs from each batch individually. \
        In the end, all these datasets were concatenated into a final dataframe we named consolidated_df_final.</h1>", unsafe_allow_html=True)
    img5b = Image.open("images/image5b.PNG")
    st.image(img5b)
    st.write("<p style='text-align: justify; font-size: 15px'>As we started building our main dataset, we begun to understand more of the potential features that were going to be included in \
        our Machine Learning models. We quickly realized that players' profile data was critical to generate predictions. Attributes such as \
        the age of a player must be relevant to our predictions.</h1>", unsafe_allow_html=True)
    img5c = Image.open("images/image5c.gif")
    st.image(img5c)
    st.write("<p style='text-align: justify; font-size: 15px'>The older you get, the most likely to get injured... Before we spun the wheels, we had to push down the brakes and head back to \
        the fbref.com website to harvest more data. This process was similar, but in this case we scraped information on a per country basis to \
        obtain each player's profile information. This yielded the following DataFrames:</h1>", unsafe_allow_html=True)
    table = pd.DataFrame(columns=['Country', 'DataFrame Name', 'Rows', 'Columns'])
    table['Country'] = ['England', 'Italy', 'Spain', 'France', 'Germany']
    table['DataFrame Name'] = ['player_data_df_england', 'player_data_df_italy', 'player_data_df_spain', 'player_data_df_france', \
        'player_data_df_germany']
    table['Rows'] = [6626,8255,7274,7354,6318]
    table['Columns'] = [15,15,15,15,15]
    table
    st.write("<p style='text-align: justify; font-size: 15px'>Once all tables were completed, those are combined into a single dataframe of all players' profiles, where we end up with a number \
        of 10,720 players. However, we only used 5,192 players since those had data available from both sources. Here is the new players_info_df:</h1>", unsafe_allow_html=True)
    img5d = Image.open("images/image5d.png")
    st.image(img5d)
    st.write("")
    st.write("<p style='text-align: justify; font-size: 15px'>The transfermarkt.com website was our source for detailed data about each player's injury history. A similar \
        scraping process to the one used with the fbref.com website was applied here. First scraping the league urls, then using \
        these league urls to scrape the team urls and then these team urls to find the player urls. Finally we used the player urls \
        to scrape the injury information for each player. This yielded a DataFrame with shape (55,216, 8) named player_injuries_df.</h1>", unsafe_allow_html=True)
    player_injuries_df = pd.read_csv('player_injuries_df.csv')
    player_injuries_df
    st.write("<p style='text-align: justify; font-size: 15px'>Additional information about the players' profile was scraped from transfermarkt.com by using the player urls.  This \
        process was done in batches of 4,000 records and it yielded the following DataFrames:</h1>", unsafe_allow_html=True)
    table2 = pd.DataFrame(columns=['DataFrame Name', 'Shape'])
    table2['DataFrame Name'] = ['player_profile_df_1', 'player_profile_df_2', 'player_profile_df_3']
    table2['Shape'] = ['(4000, 41)', '(4000, 41)', '(4000, 41)']
    table2
    st.write("<p style='text-align: justify; font-size: 15px'>This dataset contained additional information that the FBRef site did not provide. Here we found new attributes such the date \
        a player joinned a club, the date they retired, and other features we believed could be useful. However, were any of those features \
        actually used in our models? Please stay tuned...</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align: justify; font-size: 15px'>Here is the new tm_profile_df dataset after the concatenation.</h1>", unsafe_allow_html=True)
    img5f = Image.open("images/image5f.PNG")
    st.image(img5f)
    st.write("")
    st.write("<p style='text-align: justify; font-size: 15px'>The complete scraping process to get the data was done using the [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) \
        Python library.</h1>", unsafe_allow_html=True)
    
elif section == "Data Manipulation & Feature Engineering":
    st.image(imglogo, width=250)

    img6 = Image.open("images/image6.jpg")
    st.image(img6, width = 700)
   
    st.header("Merging, Cleaning and Manipulating the Data")

    st.write("<p style='text-align: justify; font-size: 15px'>This is the time when we inspected, cleaned, transformed, and merged our datasets with the ultimate goal of producing a final dataset where \
        and select a subset of input features from the final dataset in order to construct our machine learning tool. We achieved this by merging on the \
        intersection of all dataframes using the fbref_to_tm_df reference table on columns TMId(FBRef unique IDs) and TMID(TransferMarkt unique IDs) respectively. This phase \
        of the project required unbiased analysis or evaluation of how each attribute could contribute to our models as well as trial and error to experiment \
        with various methods until finding the most successful features. We needed to also avoid adding redundant variables as this could have reduced the \
        generalization capability of the model and decreased the overall accuracy. Attributes such as a player's number of minutes played could imply that the more a player \
        plays, the more likely a player is to get injured. Thus, we concluded that this feature had to be included. On the other hand, we first \
        believed that weight could have also been a key feature to maintain. However, most soccer players have to go through rigorous training and \
        stay in shape; thus, players' weights did not contribute much to our models. Additionally, our data also gave us room to reengineer some \
        features. Moreover, we created additional features from our existing dataset. Who is more likely to get injured? A goalkeeper or an attacker? \
        At first, we thought the attacker, but this may not be completely true. Again, in this stage, we were just learning and discovering trends \
        from our data. Furthermore, we created dummy variables to distinguish the positions of the players. So did the position of the player contribute \
        to our model? We will see!</h1>", unsafe_allow_html=True)
    img5e = Image.open("images/image5e.jpg")
    st.image(img5e) 
    st.write("")
    st.write("<p style='text-align: justify; font-size: 15px'>Before defining our features, we first merged all of our datasets: consolidated_df_final (FBRef match logs), players_info_df \
        (FBRef profiles), player_injuries_df (TransferMarkt injuries), and the players_info_df. We named this new dataframe as player_injuries_profile_final, \
        which yielded a shape of (159362, 75). However, this dataset changed too many times since several steps were taken as we were clean and defining \
        all features. Removing duplicates, dropping NaNs, updating the column types, and any other basic operations were applied. Most importantly, we \
        aggregated all columns at the week level. In other words, our final dataset contained all players' profile data, match logs, and injuries at the \
        week level. For example, a football player played 2 entire games within a week; then the soccer player played a total of 180 minutes. The same \
        concept arised when a player scored in multiple games within a week; if a player scored a hattrick on Tuesday and then a brace on Sunday, then a \
        single instance(row) of the data showed that this player had 5 goals. This step aggregated all column values with the groupby function and the sum() \
        operator. This was a critical step for our time series models. Likewise, we added the weeks when players did not play and filled those with 0s. \
        That is to say, if a player didn't play a certain week, then we added a row and populate all the date columns accordingly and the remaining columns \
        were filled with 0s. Additionally, we created new columns of the week and year a player gets injured as well as the week the player is released.</h1>", unsafe_allow_html=True)
    st.write("")
    st.write("<p style='text-align: justify; font-size: 15px'>This is how the dataset looked before we aggregated the dates:</h1>", unsafe_allow_html=True) 
    img14 = Image.open("images/image14.PNG")
    st.image(img14) 
    st.write("<p style='text-align: justify; font-size: 15px'>There are more new features we begun to develop as we explored our new dataset. To name a few more, we constructed new columns to highlight \
        when player's team wins, loses or draws a game. When we thought of this, it was also determined to incorporate another feature to state when a player starts \
        the game from the beginning.</h1>", unsafe_allow_html=True) 
    img13 = Image.open("images/image13.PNG")
    st.image(img13) 
    st.write("<p style='text-align: justify; font-size: 15px'>We believed competitions or tournaments where players participated could influence our model, especially when players are on international duty \
        during major tournaments such as the world qualifiers. If a football player gets injured due to international duty, this creates a battle between the club and \
        the player's national team, but we are not interested in that. We are more interested to comprehend how during a period of time players weren't too great \
        at their respective clubs, but they were outstanding in their national teams. With this being said, are players more prone to get injured if they perform better? \
        Also, are players more likely to get hurt during a world cup qualifying game since all football players may desire to play a World Cup? Additionally, the venue \
        of a game could also have an influence on players' performance and may boost the likeliness to get injured.</h1>", unsafe_allow_html=True)
    img15 = Image.open("images/image15.PNG")
    st.image(img15)
    
    st.write("<p style='text-align: justify; font-size: 15px'>Consequently, we created dummy variables to come up with new features for all competitions available and the venue.</h1>", unsafe_allow_html=True)
    img16 = Image.open("images/image16.PNG")
    st.image(img16)
    
    img17 = Image.open("images/image17.jpg")
    st.image(img17)
    st.write("<p style='text-align: justify; font-size: 15px'>After all the data manipulation and feature engineering, we produced a new dataset we named complete_final_df_all with a 1,910,255 rows and 169 \
        columns for a total of 4,588 players. We are now ready to start building our data models, but first let's take a look at the dataset. Here we \
        want to show a subset of one of Cristiano Ronaldo's best seasons during the time he lead Real Madrid to win 'La Decima' where he broke an all time \
        record and scored 17 goals in a single season for the Champions League.</h1>", unsafe_allow_html=True) 
    st.write("<p style='text-align: justify; font-size: 15px'>Feel free to scroll up, down, left, and right")
    cr7_df = pd.read_csv('dataframes_blog/df_cristiano.csv')
    cr7_df
    st.write("<p style='text-align: justify; font-size: 15px'>Here is the list of all columns from our final dataframe. All of those features were available for our time series models:</h1>", unsafe_allow_html=True)
    df_final = pd.DataFrame(columns=['Features'])
    df_final['Features'] = ['FBRefID', 'date', 'agg_week', 'agg_year', 'Injury', 'injury_week', 'injury_year', 'Min', 'Gls', 'Ast', 'PK', 'PKatt', 'Sh', 
        'SoT', 'CrdY', 'CrdR', 'Touches', 'Press', 'Tkl', 'Int', 'Blocks', 'xG', 'npxG', 'xA', 'SCA', 'GCA', 'Cmp', 'Att', 'Prog', 'Carries', 'Prog.1', 
        'Succ', 'Att.1', 'Fls', 'Fld', 'Off', 'Crs', 'TklW', 'OG', 'PKwon', 'PKcon', 'Won', 'Loss', 'Draw', 'FootAbility', 'release_week', 'was_match',
        'Serie A', 'Premier League', 'La Liga', 'Ligue 1', 'Bundesliga', 'Champions Lg', 'Europa Lg', 'FIFA World Cup', 'UEFA Nations League',
        'UEFA Euro', 'Copa América', 'Away', 'Home', 'Neutral', 'week', 'year', 'PlayerFullName', 'name', 'Position:', 'Height', 'Weight', 'Foot',
        'Birth', 'Nationality', 'Photo', 'InternationalReputation', 'Twitter', 'Instagram', 'Place of birth:', 'Citizenship:', 'cum_week', 'defender',
        'attacker', 'midfielder', 'goalkeeper', 'right_foot', 'left_foot', 'injury_count', 'cum_injury', 'age', 'unique_injury_count',
        'cum_injury_total', 'previous_injury_week', 'weeks_since_last_injury', 'injured', 'injured_in_1_week', 'injured_in_4_week',
        'injured_in_12_week', 'injured_in_26_week', 'injured_in_52_week', 'injury_count_in_1_week', 'injury_count_in_4_week',
        'injury_count_in_12_week', 'injury_count_in_26_week',  'injury_count_in_52_week', 'cum_injury_in_1_week', 'cum_injury_in_4_week',
        'cum_injury_in_12_week', 'cum_injury_in_26_week', 'cum_injury_in_52_week', 'cum_sum', 'Min_cum',  'Gls_cum', 'Ast_cum', 'PK_cum', 
        'PKatt_cum', 'Sh_cum', 'SoT_cum', 'CrdY_cum', 'CrdR_cum', 'Touches_cum', 'Press_cum', 'Tkl_cum', 'Int_cum', 'Blocks_cum', 'xG_cum', 
        'npxG_cum', 'xA_cum', 'SCA_cum', 'GCA_cum', 'Cmp_cum', 'Att_cum', 'Prog_cum', 'Carries_cum', 'Prog.1_cum', 'Succ_cum', 'Att.1_cum', 
        'Fls_cum', 'Fld_cum', 'Off_cum', 'Crs_cum', 'TklW_cum', 'OG_cum', 'PKwon_cum', 'PKcon_cum', 'Won_cum', 'Loss_cum', 'Draw_cum', 'was_match_cum', 
        'Serie A_cum', 'Premier League_cum', 'La Liga_cum', 'Ligue 1_cum', 'Bundesliga_cum', 'Champions Lg_cum', 'Europa Lg_cum', 'FIFA World Cup_cum', 
        'UEFA Nations League_cum', 'UEFA Euro_cum', 'Copa América_cum', 'Away_cum', 'Home_cum', 'Neutral_cum', 'defender_cum', 'attacker_cum', 
        'midfielder_cum', 'goalkeeper_cum', 'right_foot_cum', 'left_foot_cum', 'drop', 'last_week', 'train_split']
    # df_final['Description'] = ['Name of soccer player', 'FBRef Id', 'Date of the occurrence: game, injury or both', \
    #     'Week of the occurrence: game, injury or both', 'Year of the occurrence: game, injury or both', 'Type of injury', \
    #     'Week when injury occurred', 'Year when injury occurred', 'Minutes played', 'Goals scored or allowed', 'Completed assists', \
    #     'Penalty kicks made', 'Penalty kicks attempted', 'Shots (not including penalty kicks)', \
    #     'Shots on target (not including penalty kicks)', 'Yellow cards', 'Red cards', 'Touches in attacking penalty area', \
    #     'Passess made while under pressure of opponent', 'Number of players tackled', 'Interceptions', \
    #     'Number of times blocking the ball by standing on its path', 'Expected goals', 'Non-penalty expected goals', \
    #     'Expected assists previous to a goal', 'Two offensive actions previous to a shot', \
    #     'Two offensive actions previous to a goal', 'Passess completed', 'Passess attempted', \
    #     'Passess that move the ball at least 10 yards toward opponent goal', \
    #     'Number of times player controlled the ball with his feet', \
    #     "Carries that move the ball toward opponent's goal at least 5 yards", 'Dribbles completed successfully', \
    #     'Dribbles attempted', 'Fouls committed', 'Fouls drawn', 'Offsides', 'Crosses', \
    #     'Tackles were possession of the ball was won', 'Own goals', 'Penalty kicks won', 'Penalty kicks conceded', 'Game won', \
    #     'Game lost', 'Game draw', 'Week when player was released from injury', \
    #     'If there was a match during that week, variable = 1', 'Player height', 'Player weight', 'Player date of birth', \
    #     'Cumulative week', 'Player is a defender', 'Player is an attacker', 'Player is a midfielder', 'Player is a goalkeeper', \
    #     'Player age', 'Player is right-footed', 'Player is left-footed', 'Player number of injuries', 'Player cumulative injuries']
    df_final
    
    st.write("<p style='text-align: justify; font-size: 15px'>Did we use all features for our predictions? Of course not...</h1>", unsafe_allow_html=True)

elif section == "Visual Exploration of Data":
    st.image(imglogo, width=250)

    img7 = Image.open("images/ball.png")
    st.image(img7, width = 700)
    
    st.header('Visual Exploration of Data')
    st.write("<p style='text-align: justify; font-size: 15px'>The idea here was to execute data exploration to understand the relationships between the dependent and the independent \
        variables. The dataset contained 2 possible classes in the target variable: 0 if a player is not injured and 1 if a player is \
        injured. The target value was studied at different time windows to see how probable it was that the player would get injured in \
        the next quarter, the next semester or the next year. These classes had the following proportions:</h1>", unsafe_allow_html=True)
    img7 = Image.open("images/image7.png")
    st.image(img7)
    st.write("<p style='text-align: justify; font-size: 15px'>We can see that the dominant class is 0: when players are not injured, which makes sense because we don't expect players \
        to be injured more time than they are not injured. So our data is unbalanced which will have to be taken into account when we \
        modelling.</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align: justify; font-size: 15px'>We did some additional explorations to see if the data made 'sense'.  We wanted to see the relationship between minutes \
        played and the age of players.</h1>", unsafe_allow_html=True)
    img8 = Image.open("images/image8.png")
    st.image(img8)
    st.write("<p style='text-align: justify; font-size: 15px'>In this case it's interesting to see that there seems to be an 'optimal' age where players tend to play more minutes. \
        It looks like players between 20 and 34 years old play more minutes. This is unexpected, as we would have thougt that younger \
        players would play more minutes, but on second thought, it makes sense due to their career development.</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align: justify; font-size: 15px'>We also plotted minutes played (Min) vs the accumulated number of injuries per player (cum_injury_total).</h1>", unsafe_allow_html=True)
    img9 = Image.open("images/image9.png")
    st.image(img9)
    st.write("<p style='text-align: justify; font-size: 15px'>In this case we find a logical pattern, that player with less accumulated injuries tend to play more minutes.</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align: justify; font-size: 15px'>Additionally, we plotted player's weight (Weight) vs the accumulated number of injuries per player (cum_injury_total).</h1>", unsafe_allow_html=True)
    img10 = Image.open("images/image10.png")
    st.image(img10)
    st.write("<p style='text-align: justify; font-size: 15px'>In this case there seems to be a concentration of players between 65 kilos and 85 kilos that gets more injuries. This \
        is probably due to the fact that most players weigh in that range.</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align: justify; font-size: 15px'>We decided to plot a correlation matrix (heatmap style) to look at our whole universe of variables.</h1>", unsafe_allow_html=True)
    img11 = Image.open("images/image11.png")
    st.image(img11)
    st.write("<p style='text-align: justify; font-size: 15px'>As we can see, com_injury_total seems to have a higher positive correlation with variables like: Weight and Age.</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align: justify; font-size: 15px'>To get idea of how variables are correlated we 'zoomed-in' to just 5 variables like: 'Height', 'Weight', 'age', \
        'cum_injury_total', 'weeks_since_injury', and 'Min_cum'.</h1>", unsafe_allow_html=True)
    img12 = Image.open("images/image12.png")
    st.image(img12)
    st.write("<p style='text-align: justify; font-size: 15px'>Here we can analyze if there are any correlations between variables.  Logically, height is positively correlated with \
        weight. Being that we are analyzing active athletes, there doesn't seem to be any correlation between weight and age. We also \
        see a positive correlation between age and the cum_injury_total.</h1>", unsafe_allow_html=True)
    st.header("Deeper Data Exploration")
    st.write("<p style='text-align: justify; font-size: 15px'>The following set of tools has been developed to evaluate and understand how players' injuries evolve over time. It is reasonable to \
        assume that as players age, they are more likely to become injured. Of course, there are some players that do not get injured as much as others \
		while others get injured a lot more frequently. The following tools are intended to help us understand some of the differences \
        between these players.</h1>", unsafe_allow_html=True) 
    st.subheader("Compare Players' Injury History")
    st.write("<p style='text-align: justify; font-size: 15px'>First, we want to individually compare players' injury history. Here, you are able to select up to 3 players. This line chart exhibits a clear comparison of \
        how players start accumulating injures throughout time, where time is displayed in cumulative weeks. For this example, we are showing a model comparison on \
        3 players with similar attributes. Football fans love to compare Ronaldo and Messi. They both have had an amazing career, have scored many goals, and have broken \
        all the records, and it seems that they will continue to do so for the next few years.  However, Leo has accumulated more injuries than CR7 even though Leo has \
        played fewer games. We are including Leo and Cristiano for our comparison, and also incorporate Robert Lewandoski since he has become a scoring machine the past \
        years. It seems that Lewandoski is projected to accumulate fewer injuries than Leo. Why did Lionel Messi accumulate more injuries than Cristiano? Lionel Messi plays \
        the South American qualifiers and Copa America while Cristiano and Lewandoski play the European qualifiers and the Eurocup. Is it because of the competitions? \
        One thing to mention is that Messi is a midfielder while Cristiano and Lewandoski are strikers. May the position of the player influence the likeliness of a player \
        getting injured? We mentioned earlier that goalkeepers may not be as exposed as other positions. However, they do get injured. The following visualization will \
        help us answer this.</h1>", unsafe_allow_html=True)
    img18 = Image.open("images/image18.PNG")
    st.image(img18)
    st.subheader("Compare Cummulative Injury History According to Position")
    st.write("<p style='text-align: justify; font-size: 15px'>This tool allows you to compare the variation of injuries based on the player's position. We have previously defined 4 different positions for each player, \
        in this tool, you can compare all players' positions at the same time. When we only select the goalkeeper position, we can immediately notice that they \
        are less likely to get injured compared to other positions. Now, we can strongly say that, in fact, goalkeepers have a much lower chance of getting injured. Furthermore, \
        you may think that attackers are the ones with the higher risk of getting injured; however, this graph verifies that defenders are the ones that accumulate more injuries.</h1>", unsafe_allow_html=True)
    img19 = Image.open("images/image19.PNG")
    st.image(img19)
    st.subheader("Compare Player Injury History vs. the Average Injuries in the Position He Plays")
    st.write("<p style='text-align: justify; font-size: 15px'>The next visualization helps you compare a player's injuries through his entire career against the average injuries of players in the same position of \
        the player you select. It is crucial to understand how this player is doing compared to other players in his position. Here you can visualize where a \
        player stands with other similar players. In this case, Gareth Bale has been appointed since he is a phenomenal player who unfortunately has had had many injuries during his football career. As shown, he has \
        accumulated more injuries than the cumulative average of all the attackers from our dataset. Thus, this is a player that may continue to get injured until the \
        end of his football profession.</h1>", unsafe_allow_html=True)
    st.subheader("Compare Player Injury History vs. the Average Injuries for His Age")
    img20 = Image.open("images/image20.PNG")
    st.image(img20)
    st.write("<p style='text-align: justify; font-size: 15px'>Here, we wanted to take a very similar approach by comparing a player's injury history against the average injuries for players of the same age as the selected player. Again, \
        we used Gareth Bale as an example, and the same trend occurs where he's above the average cumulative total injuries of players his age. Bale's \
        injuries have continued to increase at a steady phase during the years. It seems that he has not been able to have a full season without injuries. He is \
        just one of those players who keeps suffering from injuries setbacks.</h1>", unsafe_allow_html=True)
    img21 = Image.open("images/image21.PNG")
    st.image(img21)
    st.subheader("Compare Player Injury History vs. the Average Player's Injuries")
    st.write("<p style='text-align: justify; font-size: 15px'>Last, this is a comparison of a single player's injuries history against the average injuries of all players. The x-axis represents the cumulative minutes \
        of all games played, and on the y-axis, the graph displays the cumulative injuries through time. Again, we chose Robert Lewandoski to be the player to be compared. As \
		shown on this graph, as he started to accumulate minutes at the beginning of his career, he wouldn't get as many injuries as the average football player. Once he \
		reaches over 40,000 minutes, he overtakes this average and starts to accumulate more injuries than the average player.</h1>", unsafe_allow_html=True)
    img22 = Image.open("images/image22.PNG")
    st.image(img22)
    
elif section == "Model Building":
    st.image(imglogo, width=250)

    img8 = Image.open("images/footballfire.jpeg")
    st.image(img8, width = 700)
    
    st.header("Model Building")
    
    st.write("""Now we are into the model building phase of the project. The first thing we need to do is to specify the target variables. In this case, \
        we are looking at historic data of players to see when injuries occured and try to use that information to anticipate when injuries are likely to happen in the future.\
            This was done by creating the target variable, whether a player got injured or not, using five different time periods:
            \n- One Week
            \n- One Month
            \n- One Quarter (3 months)
            \n- One Semester (6 months)
            \n- One Year (12 months)
            
            \nThis was done by creating the target variables in the dataset using the function below:
            """)
    with st.echo():
        
        def shift_by_time_period(df, shift_factor, column):
            df[column + '_in_' + str(shift_factor) + '_week'] = df.groupby('FBRefID')[column].shift(shift_factor*-1)
            return df

# SECTION: INJURY PREDICTION TOOL
elif section == "Injury Prediction":
    st.image(imglogo, width=250)

    imgballinfield = Image.open("images/ballinfield.jpg")
    st.image(imgballinfield, width = 700)

    st.header('Injury Prediction')

    st.write("<p style='text-align: justify; font-size: 15px'>As we have mentioned along our blog, there have been 5 previous stages before being able to do any kind of prediction. \
        Scraping of data from the web, data manipulation, feature engineering, visual exploration of data and model building, all gave \
        us the best models to predict injuries.</h1>", unsafe_allow_html=True)
    
    st.write("<p style='text-align: justify; font-size: 15px'>To start our prediction process we broke up the task into prediction horizons. We decided to predict if a player would \
        get injured during the next week, month, quarter, or year. Our model numbers showed that as our prediction was further away, \
        it was harder to predict with accuracy if a player would get injured.  The best horizon was the week horizon with an F1-score \
        of .41 using the Light Gradient Boosting Machine.</h1>", unsafe_allow_html=True)

    st.write("<p style='text-align: justify; font-size: 15px'>Once all the models were trained, they were saved into a pickle file to retrieve later. Turns out there were different \
        models in different horizons.  For example as we mentioned before the one week horizon had the Light Grdient Boosting Machine \
        as its best performing model. The 1 semester horizon had the Ada Boost Classifier as its best performing model. Producing \
        predictions was a fairly simple process after we had finished all the preceding tasks.  We loaded the models and fired up a \
        prediction for all the values in our data set.  Then we accounted for the predicted injuries that had an injury the week before. \
        That is, if the model predicted an injury the week before we would reset the next week to zero assuming that an already injured \
        player could not get injured again.  Once we had these numbers we accumulated them in a single column and accounted for the time \
        window we were predicting.  These predicted values were finally combined with the real values of the dataset to create a continuous \
        time series with a line with two colors, blue for real injuries and orange for predicted injuries. We basically decided to do \
        this to aid the viewer in detecting injuries and making inferences from them.</h1>", unsafe_allow_html=True)

    st.write("<p style='text-align: justify; font-size: 15px'>We are going to pick one of the best soccer players in the world, Neymar, as our example player.  As it turns out \
        Neymar has very interesting numbers.</h1>", unsafe_allow_html=True)

    week = Image.open('images/week.png')
    st.image(week, width = 800)

    st.write('Our system predicts that Neymar will not get injured during the next week.')

    # month = Image.open('images/month.png')
    # st.image(month, width = 800)
    
    quarter = Image.open('images/quarter.png')
    st.image(quarter, width = 800)

    st.write("<p style='text-align: justify; font-size: 15px'>However, if we look at the quarter window, we see that according to our model Neymar is going get a single injury \
        in the next 12 weeks.  This injury will happen in around 7 weeks.</h1>", unsafe_allow_html=True)

    semester = Image.open('images/semester.png')
    st.image(semester, width=800)

    st.write("<p style='text-align: justify; font-size: 15px'>The one semester prediction tells us that Neymar will only have 1 injury during the next semester.</h1>", unsafe_allow_html=True)
    
    year = Image.open('images/year.png')
    st.image(year, width=800)

    st.write("<p style='text-align: justify; font-size: 15px'>Finally, the one year prediction yields that Neymar is going to get injures 3 times in the next year.</h1>", unsafe_allow_html=True)

    st.write("<p style='text-align: justify; font-size: 15px'>Please use our interactive prediction tool which is located two sections down.</h1>", unsafe_allow_html=True)
    
elif section == "Interactive Exploration Tool (BETA)":
    st.image(imglogo, width=250)

    imgsoccer = Image.open("images/soccer.jpg")
    st.image(imgsoccer, width = 700)
    
    st.header('Interactive Exploration Tool (BETA)')
    st.write('Please feel free to try out our interactive exploration tool!')
    
    # cluster_state = st.empty()

    # @st.cache(allow_output_mutation = True)
    def load_data():
        df = dd.read_parquet('dataframes_blog/df_small.parquet')
        return df

    dataset = load_data()
        
    # Plotting Chart 1: Compare Players' Injury History

    st.subheader("Compare Players' Injury History")
    st.write('* (sample dataset used for performance purposes)')

    dataset = dataset.compute()

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

# Plotting Chart 2: Compare Cummulative Injury History According to Position
    @st.cache(allow_output_mutation = True)
    def load_data_chart2():
        df = dd.read_parquet('dataframes_blog/df_pos.parquet')
        return df

    df_pos = load_data_chart2()
    
    st.subheader("Compare Cummulative Injury History According to Position")
    st.write('* (sample dataset used for performance purposes)')
    
    df_pos = df_pos.compute()

    positions = ['attacker', 'defender', 'goalkeeper', 'midfielder']

    selected_position = st.multiselect('Choose Positions to show:', positions, positions)
    
    result = pd.DataFrame([])
    result['cum_week'] = df_pos['cum_week']

    for pos in selected_position:
        result[pos] = df_pos[pos]
        
    base = alt.Chart(result).encode(x='cum_week:Q')

    chart2 = alt.layer(base.mark_line(color='red').encode(y='attacker:Q'), base.mark_line(color='orange').encode(y='defender:Q'), \
                base.mark_line(color='green').encode(y='goalkeeper:Q'), alt.layer(base.mark_line(color='blue').encode(y='midfielder:Q'))). \
                properties(width=800, height=300)
    
    st.altair_chart(chart2, use_container_width=False)

# Plotting Chart 3:  Compare Player Injury History vs. the Average Injuries in the Position He Plays

    st.subheader("Compare Player Injury History vs. the Average Injuries in the Position He Plays")
    st.write('* (sample dataset used for performance purposes)')
    
    player = st.selectbox('Player Name (type or choose):',sorted_unique_player)
    picked_player_pos = dataset[dataset['name'] == player]['position'].iloc[0]
    st.write(player + " plays as " + picked_player_pos + "!!!")
    
    @st.cache(allow_output_mutation=True)
    def chart3(player, df):
        df_player = df[df['name'] == player][['cum_week', 'name', 'cum_injury_total']]
        player_max_cum_week = df_player['cum_week'].max()
        
        df_avg_position = df[df['position'] == picked_player_pos]
        df_avg_position = df_avg_position[df_avg_position['cum_week'] <= player_max_cum_week]
        df_avg_position = df_avg_position.groupby('cum_week').mean().reset_index()[['cum_week', 'cum_injury_total']]
        df_avg_position['name'] = picked_player_pos+'s avg accum. injuries'

        df_player_vs_avg = pd.concat([df_player, df_avg_position])

        chart3 = alt.Chart(df_player_vs_avg).mark_line().encode(x=alt.X('cum_week:Q'), y='cum_injury_total:Q', color='name'). \
            properties(width=800, height=300)
        
        return chart3

    chart3_output = copy.deepcopy(chart3(player, dataset))

    st.altair_chart(chart3_output, use_container_width=False)

# Plotting Chart 4: Compara Player Injury History vs. the Average Injuries for His Age
    
    st.subheader("Compare Player Injury History vs. the Average Injuries for His Age")
    st.write('* (player ages are updated with the latest data we have)')
    st.write('* (sample dataset used for performance purposes)')

    player2 = st.selectbox("Player's Name (type or choose):",sorted_unique_player)
    
    picked_player_age_start = dataset[dataset['name'] == player2]['age'].min()
    picked_player_age_now = dataset[dataset['name'] ==player2]['age'].max()
    
    picked_player = dataset[dataset['name'] == player2][['name', 'age', 'cum_injury_total']]
    
    st.write(player2 + " has data since the age of " + str(int(picked_player_age_start)) + ", and he is now " + \
        str(int(picked_player_age_now)) + " years old!!!")

    @st.cache(allow_output_mutation=True)
    def chart4(player, df):
        df_player2 = df[df['name'] == player2][['name', 'age', 'cum_injury_total']]

        picked_player_max_age = df_player2['age'].max()

        df_avg_age = df[['cum_week', 'name', 'age', 'cum_injury_total']]
        df_avg_age = df_avg_age[df_avg_age['age'] <= picked_player_max_age]
        df_avg_age = df_avg_age.groupby('age').mean().reset_index()[['age', 'cum_injury_total']]
        df_avg_age['name'] = 'avg cum_injury_total'

        df_player_vs_avg_age = pd.concat([df_player2, df_avg_age])

        chart4 = alt.Chart(df_player_vs_avg_age).mark_line().encode(x=alt.X('age:Q'), y='cum_injury_total:Q', color='name'). \
            properties(width=800, height=300)

        return chart4

    chart4_output = copy.deepcopy(chart4(player, dataset))
        
    st.altair_chart(chart4_output, use_container_width=False)

# Plotting Chart 5 Compare Player Injury History vs. the Average Player's Injuries
    st.subheader("Compare Player Injury History vs. the Average Player's Injuries")
    st.write('* Player ages are updated with the latest data we have *')
    st.write('* (sample dataset used for performance purposes)')

    player5 = st.selectbox("Name (type or choose):",sorted_unique_player)

    @st.cache(allow_output_mutation=True)
    def chart5(player, df):  
        df_picked_player = df[df['name'] == player5][['cum_week', 'name', 'Min', 'cum_injury_total']]
        df_picked_player['cum_Min'] = df_picked_player['Min'].cumsum()

        cum_Min_max = df_picked_player['cum_Min'].max()

        df_avg_min = df[['cum_week', 'name', 'Min', 'cum_injury_total']]
        df_avg_min['cum_Min'] = df_avg_min.groupby(by=['name'])['Min'].cumsum()
        df_avg_min = df_avg_min.groupby('cum_week').mean().reset_index()
        df_avg_min['name'] = 'avg of all players'

        df_avg_min = df_avg_min[df_avg_min['cum_Min'] <= cum_Min_max]

        df_picked_player.drop_duplicates(inplace=True)
        df_avg_min.drop_duplicates(inplace=True)

        df_player_vs_avg_min = pd.concat([df_picked_player, df_avg_min])

        chart5 = alt.Chart(df_player_vs_avg_min).mark_line().encode(x=alt.X('cum_Min:Q'), y='cum_injury_total:Q', color='name'). \
            properties(width=800, height=300)

        return chart5

    chart5_output = copy.deepcopy(chart5(player, dataset))
        
    st.altair_chart(chart5_output, use_container_width=False)

elif section == "Interactive Injury Prediction Tool (BETA)":
    st.image(imglogo, width=250)

    imgfield = Image.open("images/fields.jpg")
    st.image(imgfield, width = 700)
    
    st.header("Interactive Injury Prediction Tool (BETA)")
    st.write('Please feel free to try out our interactive prediction tool!')

    st.write('* (sample dataset used for performance purposes)')

    



else:
    st.header("Conclusion, Challenges, and Future Work")
    st.markdown("<h1 style='text-align: center; color: red;'>Some title</h1>", unsafe_allow_html=True)

st.write("<p style='text-align: justify; font-size: 15px'>
.</h1>", unsafe_allow_html=True)

    st.subheader("Conclusion")
    st.write("<p style='text-align: justify; font-size: 15px'>In today’s football, players compete more and moreover a single year. Apart from playing more than one \
        tournament for their clubs, who own the rights for the players, the most talented players must attend \
        international duties with their national teams, as well as friendly matches with both their clubs and \
        countries. It is normal to see footballers get injured from time to time since this sport requires physical \
        demand and players are constantly being tackled and exposed to full body or kick collisions. Additionally, \
        players have to go through rigorous training and do not have much time to rest. Our goal was to create \
        a machine learning model that could predict when a player will get injured......</h1>", unsafe_allow_html=True)

    st.write("<p style='text-align: justify; font-size: 15px'>The project has been a great experience for all of us. Not only because we were challenged to come up \
        with a data science project that could be applied to the real world and also learned a great number of \
        new tools we previously did not have any knowledge about, but most importantly because this capstone \
        gave all of us the opportunity to work with one another as data scientists. We all faced many challenges, \
        encountered multiple roadblocks, stayed up long hours, but we also discovered new capabilities together \
        and supported each other at all times. We all agree that this has been the greatest takeaway from the \
        course and even the entire program. It was definitely not a simple task to complete .....</h1>", unsafe_allow_html=True)")
    
    st.subheader("Challenges")
    st.write("")
    st.write("**Data Scrapping**")
    st.write("<p style='text-align: justify; font-size: 15px'>There was a considerable number of roadblocks in data scrapping, and we did not envision the challenges before we \
        proceeded with the harvesting of data. The process of scrapping data from the web was a challenge for most of us because \
        of multiple reasons. First of all, we were not convinced on what data to scrape for our models since we wanted to build a \
        time series model. On top of that, we were looking at more than 3 sources at first. Second, there were many players available \
        and a good number of attributes to choose from the websites. However, many of those attributes weren’t available for certain \
        leagues or players, and when they were available, it seems that they were only obtainable for certain players. Also, those new \
        features were not available for all the years, this is probably because this is new data that just started to being recorded \
        recently. Third, scraping the data from the web required high computation and a great amount of memory capacity. We tried a few \
        strategies to work around it. We attempted to use AWS, but we also faced many challenges with the installation as well as \
        making the connection to the server; AWS preserved losing connection which resulted in errors in the middle of our data pulling \
        process. On the other hand, we split the list of players to be scrapped in multiple batches between the three of us. This not only \
        required time since we were running scripts overnight but also, required all of us to continue checking on the progress and \
        communicating with each other to ensure the scrapping was working or finalized. Additionally, when we first completed scrapping \
        the data from our sources, we realized that something was missed so we needed to go back to improve our API and repeat the same \
        process more than one time. We even bumped into a few IP blocks after a high number of requests from the same IP address. In \
        the end, we managed to scrape all the data that was required to build our models by working cooperatively.</h1>", unsafe_allow_html=True)
    st.write("")
    st.write("**GitHub in Visual Studio Code**")
    st.write("<p style='text-align: justify; font-size: 15px'>First, it took some effort to locally install and integrate VS Code and GitHub together. After the installation finished,\
        and the repository was set and authenticated, we were capable to begin working in parallel, which we all enjoyed. However, we did \
        undergo some minor problems when submitting new changes and pulling new requests. This mainly occurred because we failed to \
        communicate which notebooks were being updated, as well as not pulling on time when needed to. This resulted in many conflicts \
        that were not straightforward to solve, which even cause resetting the repositories in some cases. Once we were all accustomed to it, \
        we did not face major problems.</h1>", unsafe_allow_html=True)
    st.write("")
    st.write("**Amazon Web Services**")
    st.write("<p style='text-align: justify; font-size: 15px'>In addition to using VS Code to work with our GitHub repository, we incorporated Amazon Web Services so that we can attempt\
        to run scripts that required a great amount of memory usage. The process of setting AWS was lengthy, and although we found guidelines\
        on how to install and integrate AWS into our machines, it was still confusing to deal with it. Once AWS was installed and running, \
        all of us started to experience problems with the connection during the web scrapping phase. This occurred regularly, thus we \
        decided to avoid employing AWS for the data scrapping..</h1>", unsafe_allow_html=True)
    st.write("")
    st.write("**Time Series Model**")
    st.write("<p style='text-align: justify; font-size: 15px'>Building our time series dataset was not an easy task. It required a great amount of time to examine and debate how the model \
        was going to be built as well as what features had to be included, and what data instances we had to drop. It took multiple attempts \
        to construct the desired data frame, and it also required us to investigate the actual machine learning algorithms we were going to employ \
        before we prepare the final dataset..</h1>", unsafe_allow_html=True)
    st.write("")
    st.write("**StreamLit**")
    st.write("<p style='text-align: justify; font-size: 15px'>This python library allowed us to build our blog for the audience. It did not take a great amount of research since \
        it was fairly easy to understand compared to other libraries we have used before, and also Streamlit integrated amazingly \
        well with other libraries such Pandas or Altair. However, the challenge was when building our Players’ comparison tool. \
        Even though we achieved to develop our tools and displayed them through Streamlit, it was taken a very long time for the \
        custom apps to run because of the size of our main dataset and the magnitude of our models. We wanted to avoid creating \
        frustration between our users, so we decided to take a different approach. First, we converted our main data frame from \
        a CSVfile to a parquet file. This significantly reduced the weight of the dataset; nevertheless, the tool barely improve.</h1>", unsafe_allow_html=True)

    st.subheader("Future Work")
    
    st.write("<p style='text-align: justify; font-size: 15px'>We have collected a significant amount of data that contains players’ \
    profiles, stats, match logs, injuries, \
    and more. With this data, we have created robust models that can predict players' injuries. However, \
    there’s still room for improvement in our results if were to add more data. This may be possible by \
    accessing biometric data which relates to the measurement of players' physical features and \
    characteristics, GPS information of players during games and training, and even weather data since this \
    may affect players' performance. Most of the data that we have collected comes from what was publicly \
    available to us, thus, we do not own any information on what the footballer does before and after the \
    actual matches. Nowadays, players are playing more games per season than ever, they barely have time \
    to recover, and also have to go through rigorous training. In fact, there are many players that get injured \
    during training. On top of that, they also have to constantly travel which can also contribute to \
    accumulating more fatigue over time. Unfortunately, accessing biometrics and GPS data to evaluate the \
    rigorous training and collect features that are disregarded could improve our model. Nevertheless, this \
    was not possible due to the confidentiality of data and the time that it may take to incorporate new \
    information. In addition, accessing the number of hours players rest before and after may be challenging \
    to obtain and even raise some ethical issues.</h1>", unsafe_allow_html=True)

   


     