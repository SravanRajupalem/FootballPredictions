import streamlit as st
from PIL import Image
import pandas as pd
import altair as alt
from pathlib import Path
import os
import requests
import copy
import dask.dataframe as dd

imglogo = Image.open("images/logo.png")

section = st.sidebar.selectbox("Sections", ("Introduction", "Scraping the Web for Data", "Data Manipulation & Feature Engineering", 
    "Visual Exploration of Data", "Model Building", "Injury Prediction", "Interactive Exploration Tool (BETA)", 
    "Interactive Injury Prediction Tool (BETA)", "Conclusions and Future Work"))

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
    imgcorner = Image.open("images/corner.jpg")
    st.image(imgcorner, width=700)

    imglogo = Image.open("images/logo.png")
    st.image(imglogo, width=250)
    st.header('Scraping the Web for Data')
    st.write("We hunted the web to get the most information we could about soccer players and matches.  After scanning several \
        options our runners up due to the completeness of their data were:  fbref.com and transfermarkt.com.")
    img = Image.open("images/image1.png")
    img2 = Image.open("images/image2.png")
    img3 = Image.open("images/image3.png")
    st.image(img)
    st.image(img2)
    st.image(img3)
    st.write("The first major decision was that we would only get information from the five most competitive soccer leagues in \
        the world: Premier League (England), La Liga (Spain), Bundesliga (Germany), Ligue 1 (France) and the Serie A (Italy). \
        The reason for this decision was that we thought that these leagues would have better player documentation.")
    img4 = Image.open("images/image4.png")
    st.image(img4)
    st.write("From FBRef.com we first scraped urls from the big 5 European leagues. With that base, we again scraped the website \
        for all the seasons for each league. Then we scraped the players' urls from each of all available seasons of the top 5. \
        This operation yielded a list of 78,959 unique records. Those embedded urls contained an identifier (FBRefID) for each of the \
        19,572 players from this fbref.com. Moreover, since we intended to scrape complementary players' data from the TransferMarkt \
        website, we decided to only pull data for the players whose information was available on both sites. ")    
    st.write("Next, we had to use a handy mapping dataset called fbref_to_tm_mapping that links the websites' unique identifiers \
        FBRefID and TMID (TransferMarkt ID), which we downloaded from [Jason Zickovic](https://github.com/JaseZiv/worldfootballR_data/tree/master/raw-data/fbref-tm-player-mapping/output) \
        (kudos!). Via string conversions and splitting we extracted the FBRefID's from the generated list and decided to only scrape \
        the match logs for those players from the FBRef site.")
    img5 = Image.open("images/image5.png")
    st.image(img5)
    st.write("This effort helped us reduce a significant amount of memory usage when performing the data scrapping given that only 5,192 \
        players had attainable data from both sites. Now we can execute another pull, but this time we obtained a list of 51,196 the complete \
        match logs urls of all the consolidated players")
    img5a = Image.open("images/image5a.PNG")
    st.image(img5a)
    st.write("This is where the real data scraping of the players' match logs begun. The extraction of all players matches required high \
        computation; thus, our team divided the data extraction in multiple batches, where we extracted the match logs from each batch individually. \
        In the end, all these datasets were concatenated into a final dataframe we named consolidated_df_final.")
    img5b = Image.open("images/image5b.PNG")
    st.image(img5b)
    st.write("As we started building our main dataset, we begun to understand more of the potential features that were going to be included in \
        our Machine Learning models. We quickly realized that players' profile data was critical to generate predictions. Attributes such as \
        the age of a player must be relevant to our predictions.")
    img5c = Image.open("images/image5c.gif")
    st.image(img5c)
    st.write("The older you get, the most likely to get injured... Before we spun the wheels, we had to push down the brakes and head back to \
        the fbref.com website to harvest more data. This process was similar, but in this case we scraped information on a per country basis to \
        obtain each player's profile information. This yielded the following DataFrames:")
    table = pd.DataFrame(columns=['Country', 'DataFrame Name', 'Rows', 'Columns'])
    table['Country'] = ['England', 'Italy', 'Spain', 'France', 'Germany']
    table['DataFrame Name'] = ['player_data_df_england', 'player_data_df_italy', 'player_data_df_spain', 'player_data_df_france', \
        'player_data_df_germany']
    table['Rows'] = [6626,8255,7274,7354,6318]
    table['Columns'] = [15,15,15,15,15]
    table
    st.write("Once all tables were completed, those are combined into a single dataframe of all players' profiles, where we end up with a number \
        of 10,720 players. However, we only used 5,192 players since those had data available from both sources. Here is the new players_info_df:")    
    img5d = Image.open("images/image5d.png")
    st.image(img5d)
    st.write("")
    st.write("The transfermarkt.com website was our source for detailed data about each player's injury history. A similar \
        scraping process to the one used with the fbref.com website was applied here. First scraping the league urls, then using \
        these league urls to scrape the team urls and then these team urls to find the player urls. Finally we used the player urls \
        to scrape the injury information for each player. This yielded a DataFrame with shape (55,216, 8) named player_injuries_df.")
    player_injuries_df = pd.read_csv('player_injuries_df.csv')
    player_injuries_df
    st.write("Additional information about the players' profile was scraped from transfermarkt.com by using the player urls.  This \
        process was done in batches of 4,000 records and it yielded the following DataFrames:")
    table2 = pd.DataFrame(columns=['DataFrame Name', 'Shape'])
    table2['DataFrame Name'] = ['player_profile_df_1', 'player_profile_df_2', 'player_profile_df_3']
    table2['Shape'] = ['(4000, 41)', '(4000, 41)', '(4000, 41)']
    table2
    st.write("This dataset contained additional information that the FBRef site did not provide. Here we found new attributes such the date \
        a player joinned a club, the date they retired, and other features we believed could be useful. However, were any of those features \
        actually used in our models? Please stay tuned...")
    st.write("Here is the new tm_profile_df dataset after the concatenation.")
    img5f = Image.open("images/image5f.PNG")
    st.image(img5f)
    st.write("")
    st.write("The complete scraping process to get the data was done using the [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) \
        Python library.")
    
elif section == "Data Manipulation & Feature Engineering":
    
    img6 = Image.open("images/image6.jpg")
    st.image(img6, width = 700)

    
    st.image(imglogo, width=250)
    st.header("Merging, Cleaning and Manipulating the Data")

    st.write("This is the time when we inspected, cleaned, transformed, and merged our datasets with the ultimate goal of producing a final dataset where \
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
        to our model? We will see!")
    img5e = Image.open("images/image5e.jpg")
    st.image(img5e) 
    st.write("")
    st.write("Before defining our features, we first merged all of our datasets: consolidated_df_final (FBRef match logs), players_info_df \
        (FBRef profiles), player_injuries_df (TransferMarkt injuries), and the players_info_df. We named this new dataframe as player_injuries_profile_final, \
        which yielded a shape of (159362, 75). However, this dataset changed too many times since several steps were taken as we were clean and defining \
        all features. Removing duplicates, dropping NaNs, updating the column types, and any other basic operations were applied. Most importantly, we \
        aggregated all columns at the week level. In other words, our final dataset contained all players' profile data, match logs, and injuries at the \
        week level. For example, a football player played 2 entire games within a week; then the soccer player played a total of 180 minutes. The same \
        concept arised when a player scored in multiple games within a week; if a player scored a hattrick on Tuesday and then a brace on Sunday, then a \
        single instance(row) of the data showed that this player had 5 goals. This step aggregated all column values with the groupby function and the sum() \
        operator. This was a critical step for our time series models. Likewise, we added the weeks when players did not play and filled those with 0s. \
        That is to say, if a player didn't play a certain week, then we added a row and populate all the date columns accordingly and the remaining columns \
        were filled with 0s. Additionally, we created new columns of the week and year a player gets injured as well as the week the player is released.")
    st.write("")
    st.write("This is how the dataset looked before we aggregated the dates:") 
    img14 = Image.open("images/image14.PNG")
    st.image(img14) 
    st.write("There are more new features we begun to develop as we explored our new dataset. To name a few more, we constructed new columns to highlight \
        when player's team wins, loses or draws a game. When we thought of this, it was also determined to incorporate another feature to state when a player starts \
        the game from the beginning")
    img13 = Image.open("images/image13.PNG")
    st.image(img13) 
    st.write("We believed competitions or tournaments where players participated could influence our model, especially when players are on international duty \
        during major tournaments such as the world qualifiers. If a football player gets injured due to international duty, this creates a battle between the club and \
        the player's national team, but we are not interested in that. We are more interested to comprehend how during a period of time players weren't too great \
        at their respective clubs, but they were outstanding in their national teams. With this being said, are players more prone to get injured if they perform better? \
        Also, are players more likely to get hurt during a world cup qualifying game since all football players may desire to play a World Cup? Additionally, the venue \
        of a game could also have an influence on players' performance and may boost the likeliness to get injured.")
    img15 = Image.open("images/image15.PNG")
    st.image(img15)
    
    st.write("Consequently, we created dummy variables to come up with new features for all competitions available and the venue.")
    img16 = Image.open("images/image16.PNG")
    st.image(img16)
    
    img17 = Image.open("images/image17.jpg")
    st.image(img17)
    st.write("After all the data manipulation and feature engineering, we produced a new dataset we named complete_final_df_all with a 1,910,255 rows and 169 \
        columns for a total of 4,588 players. We are now ready to start building our data models, but first let's take a look at the dataset. Here we \
        want to show a subset of one of Cristiano Ronaldo's best seasons during the time he lead Real Madrid to win 'La Decima' where he broke an all time \
        record and scored 17 goals in a single season for the Champions League.") 
    st.write("Feel free to scroll up, down, left, and right")
    cr7_df = pd.read_csv('dataframes_blog/df_cristiano.csv')
    cr7_df
    st.write("Here is the list of all columns from our final dataframe. All of those features were available for our time series models:")
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
    
    st.write("Did we use all features for our predictions? Of course not...")

elif section == "Visual Exploration of Data":

    img7 = Image.open("images/ball.png")
    st.image(img7, width = 700)

    st.image(imglogo, width=250)
    
    st.header('Visual Exploration of Data')
    st.write("The idea here was to execute data exploration to understand the relationships between the dependent and the independent \
        variables. The dataset contained 2 possible classes in the target variable: 0 if a player is not injured and 1 if a player is \
        injured. The target value was studied at different time windows to see how probable it was that the player would get injured in \
        the next quarter, the next semester or the next year. These classes had the following proportions:")
    img7 = Image.open("images/image7.png")
    st.image(img7)
    st.write("We can see that the dominant class is 0: when players are not injured, which makes sense because we don't expect players \
        to be injured more time than they are not injured. So our data is unbalanced which will have to be taken into account when we \
        modelling.")
    st.write("We did some additional explorations to see if the data made 'sense'.  We wanted to see the relationship between minutes \
        played and the age of players.")
    img8 = Image.open("images/image8.png")
    st.image(img8)
    st.write("In this case it's interesting to see that there seems to be an 'optimal' age where players tend to play more minutes. \
        It looks like players between 20 and 34 years old play more minutes. This is unexpected, as we would have thougt that younger \
        players would play more minutes, but on second thought, it makes sense due to their career development.")
    st.write("We also plotted minutes played (Min) vs the accumulated number of injuries per player (cum_injury_total).")
    img9 = Image.open("images/image9.png")
    st.image(img9)
    st.write("In this case we find a logical pattern, that player with less accumulated injuries tend to play more minutes.")
    st.write("Additionally, we plotted player's weight (Weight) vs the accumulated number of injuries per player (cum_injury_total).")
    img10 = Image.open("images/image10.png")
    st.image(img10)
    st.write("In this case there seems to be a concentration of players between 65 kilos and 85 kilos that gets more injuries. This \
        is probably due to the fact that most players weigh in that range.")
    st.write("We decided to plot a correlation matrix (heatmap style) to look at our whole universe of variables.")
    img11 = Image.open("images/image11.png")
    st.image(img11)
    st.write("As we can see, com_injury_total seems to have a higher positive correlation with variables like: Weight and Age.")
    st.write("To get idea of how variables are correlated we 'zoomed-in' to just 5 variables like: 'Height', 'Weight', 'age', \
        'cum_injury_total', 'weeks_since_injury', and 'Min_cum'.")
    img12 = Image.open("images/image12.png")
    st.image(img12)
    st.write("Here we can analyze if there are any correlations between variables.  Logically, height is positively correlated with \
        weight. Being that we are analyzing active athletes, there doesn't seem to be any correlation between weight and age. We also \
        see a positive correlation between age and the cum_injury_total.")
    st.header("Deeper Data Exploration")
    st.write("The following set of tools has been developed to evaluate and understand how players' injuries evolve over time. It is reasonable to \
        assume that as players age, they are more likely to become injured. Of course, there are some players that do not get injured as much as others \
		while others get injured a lot more frequently. The following tools are intended to help us understand some of the differences \
        between these players.") 
    st.subheader("Compare Players' Injury History")
    st.write("First, we want to individually compare players' injury history. Here, you are able to select up to 3 players. This line chart exhibits a clear comparison of \
        how players start accumulating injures throughout time, where time is displayed in cumulative weeks. For this example, we are showing a model comparison on \
        3 players with similar attributes. Football fans love to compare Ronaldo and Messi. They both have had an amazing career, have scored many goals, and have broken \
        all the records, and it seems that they will continue to do so for the next few years.  However, Leo has accumulated more injuries than CR7 even though Leo has \
        played fewer games. We are including Leo and Cristiano for our comparison, and also incorporate Robert Lewandoski since he has become a scoring machine the past \
        years. It seems that Lewandoski is projected to accumulate fewer injuries than Leo. Why did Lionel Messi accumulate more injuries than Cristiano? Lionel Messi plays \
        the South American qualifiers and Copa America while Cristiano and Lewandoski play the European qualifiers and the Eurocup. Is it because of the competitions? \
        One thing to mention is that Messi is a midfielder while Cristiano and Lewandoski are strikers. May the position of the player influence the likeliness of a player \
        getting injured? We mentioned earlier that goalkeepers may not be as exposed as other positions. However, they do get injured. The following visualization will \
        help us answer this.")
    img18 = Image.open("images/image18.PNG")
    st.image(img18)
    st.subheader("Compare Cummulative Injury History According to Position")
    st.write("This tool allows you to compare the variation of injuries based on the player's position. We have previously defined 4 different positions for each player, \
        in this tool, you can compare all players' positions at the same time. When we only select the goalkeeper position, we can immediately notice that they \
        are less likely to get injured compared to other positions. Now, we can strongly say that, in fact, goalkeepers have a much lower chance of getting injured. Furthermore, \
        you may think that attackers are the ones with the higher risk of getting injured; however, this graph verifies that defenders are the ones that accumulate more injuries.")
    img19 = Image.open("images/image19.PNG")
    st.image(img19)
    st.subheader("Compare Player Injury History vs. the Average Injuries in the Position He Plays")
    st.write("The next visualization helps you compare a player's injuries through his entire career against the average injuries of players in the same position of \
        the player you select. It is crucial to understand how this player is doing compared to other players in his position. Here you can visualize where a \
        player stands with other similar players. In this case, Gareth Bale has been appointed since he is a phenomenal player who unfortunately has had had many injuries during his football career. As shown, he has \
        accumulated more injuries than the cumulative average of all the attackers from our dataset. Thus, this is a player that may continue to get injured until the \
        end of his football profession.")
    st.subheader("Compare Player Injury History vs. the Average Injuries for His Age")
    img20 = Image.open("images/image20.PNG")
    st.image(img20)
    st.write("Here, we wanted to take a very similar approach by comparing a player's injury history against the average injuries for players of the same age as the selected player. Again, \
        we used Gareth Bale as an example, and the same trend occurs where he's above the average cumulative total injuries of players his age. Bale's \
        injuries have continued to increase at a steady phase during the years. It seems that he has not been able to have a full season without injuries. He is \
        just one of those players who keeps suffering from injuries setbacks.")
    img21 = Image.open("images/image21.PNG")
    st.image(img21)
    st.subheader("Compare Player Injury History vs. the Average Player's Injuries")
    st.write("Last, this is a comparison of a single player's injuries history against the average injuries of all players. The x-axis represents the cumulative minutes \
        of all games played, and on the y-axis, the graph displays the cumulative injuries through time. Again, we chose Robert Lewandoski to be the player to be compared. As \
		shown on this graph, as he started to accumulate minutes at the beginning of his career, he wouldn't get as many injuries as the average football player. Once he \
		reaches over 40,000 minutes, he overtakes this average and starts to accumulate more injuries than the average player.")
    img22 = Image.open("images/image22.PNG")
    st.image(img22)
    
elif section == "Model Building":
    img8 = Image.open("images/footballfire.jpeg")
    st.image(img8, width = 700)

    st.image(imglogo, width=250)
    st.header("Model Building")

# SECTION: INJURY PREDICTION TOOL
elif section == "Injury Prediction":
    st.header('Injury Prediction')

    
elif section == "Interactive Exploration Tool (BETA)":
    st.header('Interactive Exploration Tool (BETA)')
    
    # cluster_state = st.empty()

    @st.cache(allow_output_mutation = True)
    def load_data():
        # df = dd.read_parquet('dataframes_blog/dataset_for_model_final.parquet') #, storage_options={"anon":True}, blocksize="16 MiB")
        df = dd.read_parquet('dataframes_blog/df_small.parquet')
        return df

    dataset = load_data()
        
    # Plotting Chart 1: Compare Players' Injury History

    st.subheader("Compare Players' Injury History")
    st.write(dataset['name'])

    # sorted_unique_player = dataset['name'].sort_values().unique()
    # player1 = st.selectbox('Player 1 Name (type or choose):',sorted_unique_player)
    # player2 = st.selectbox('Player 2 Name (type or choose):',sorted_unique_player)
    # player3 = st.selectbox('Player 3 Name (type or choose):',sorted_unique_player)
    
    # @st.cache(allow_output_mutation=True)
    # def chart1(player1, player2, player3): 
    #     df1_1 = dataset[dataset['name'] == player1][['cum_week', 'name', 'cum_injury_total']]
    #     df1_2 = dataset[dataset['name'] == player2][['cum_week', 'name', 'cum_injury_total']]
    #     df1_3 = dataset[dataset['name'] == player3][['cum_week', 'name', 'cum_injury_total']]

    #     df = pd.concat([df1_1, df1_2, df1_3])
    
    #     chart1 = alt.Chart(df).mark_line().encode(x=alt.X('cum_week:Q', axis=alt.Axis(labelAngle=0)), y='cum_injury_total:Q', color='name'). \
    #         properties(width=800, height=300)

    #     return chart1
    
    # chart1_output = copy.deepcopy(chart1(player1, player2, player3))
    # st.altair_chart(chart1_output, use_container_width=False)

# # Plotting Chart 2: Compare Cummulative Injury History According to Position
#     st.subheader("Compare Cummulative Injury History According to Position")
    
#     dataset.loc[dataset['attacker'] == 1, 'position'] = 'attacker'
#     dataset.loc[dataset['midfielder'] == 1, 'position'] = 'midfielder'
#     dataset.loc[dataset['defender'] == 1, 'position'] = 'defender'
#     dataset.loc[dataset['goalkeeper'] == 1, 'position'] = 'goalkeeper'

#     df = dataset[['cum_week', 'name', 'position', 'cum_injury_total']]
#     sorted_unique_position = dataset['position'].dropna().sort_values().unique()
#     pos = st.multiselect('Positions',sorted_unique_position, sorted_unique_position)
#     df_pos = pd.DataFrame([])
#     for p in pos:
#         df_pos = pd.concat([df_pos, df[df['position'] == p]], ignore_index=True)
    
#     df_pos['attacker'] = 0
#     df_pos['defender'] = 0
#     df_pos['goalkeeper'] = 0
#     df_pos['midfielder'] = 0
#     df_pos.loc[df_pos['position'] == 'attacker', 'attacker'] = df_pos['cum_injury_total']
#     df_pos.loc[df_pos['position'] == 'defender', 'defender'] = df_pos['cum_injury_total']
#     df_pos.loc[df_pos['position'] == 'goalkeeper', 'goalkeeper'] = df_pos['cum_injury_total']
#     df_pos.loc[df_pos['position'] == 'midfielder', 'midfielder'] = df_pos['cum_injury_total']
#     df_pos = df_pos.groupby('cum_week').sum().reset_index()
#     base = alt.Chart(df_pos).encode(x='cum_week:Q')
#     chart2 = alt.layer(base.mark_line(color='red').encode(y='attacker'), base.mark_line(color='orange').encode(y='defender'), \
#         base.mark_line(color='green').encode(y='goalkeeper'), alt.layer(base.mark_line(color='blue').encode(y='midfielder'))). \
#         properties(width=800, height=300)
#     st.altair_chart(chart2, use_container_width=False)

# # Plotting Chart 3:  Compare Player Injury History vs. the Average Injuries in the Position He Plays

#     st.subheader("Compare Player Injury History vs. the Average Injuries in the Position He Plays")

#     player = st.selectbox('Player Name (type or choose):',sorted_unique_player)
    
#     picked_player_pos = dataset[dataset['name'] == player]['position'].iloc[0]
#     st.write(player + " plays as " + picked_player_pos + "!!!")

#     df_player = dataset[dataset['name'] == player][['cum_week', 'name', 'cum_injury_total']]

#     player_max_cum_week = df_player['cum_week'].max()

#     df_avg_position = dataset[dataset['position'] == picked_player_pos]
#     df_avg_position = df_avg_position[df_avg_position['cum_week'] <= player_max_cum_week]
#     df_avg_position = df_avg_position.groupby('cum_week').mean().reset_index()[['cum_week', 'cum_injury_total']]
#     df_avg_position['name'] = picked_player_pos+'s avg accum. injuries'

#     df_player_vs_avg = pd.concat([df_player, df_avg_position])

#     chart3 = alt.Chart(df_player_vs_avg).mark_line().encode(x=alt.X('cum_week:Q'), y='cum_injury_total:Q', color='name'). \
#         properties(width=800, height=300)
#     st.altair_chart(chart3, use_container_width=False)

# # Plotting Chart 4: Compara Player Injury History vs. the Average Injuries for His Age
    
#     st.subheader("Compare Player Injury History vs. the Average Injuries for His Age")
#     st.write('* Player ages are updated with the latest data we have *')

#     player2 = st.selectbox("Player's Name (type or choose):",sorted_unique_player)
    
#     picked_player_age_start = dataset[dataset['name'] == player2]['age'].min()
#     picked_player_age_now = dataset[dataset['name'] ==player2]['age'].max()
    
#     picked_player = dataset[dataset['name'] == player2][['name', 'age', 'cum_injury_total']]
    
#     st.write(player2 + " has data since the age of " + str(int(picked_player_age_start)) + ", and he is now " + \
#         str(int(picked_player_age_now)) + " years old!!!")

#     df_player2 = dataset[dataset['name'] == player2][['name', 'age', 'cum_injury_total']]

#     picked_player_max_age = df_player2['age'].max()

#     df_avg_age = dataset[['cum_week', 'name', 'age', 'cum_injury_total']]
#     df_avg_age = df_avg_age[df_avg_age['age'] <= picked_player_max_age]
#     df_avg_age = df_avg_age.groupby('age').mean().reset_index()[['age', 'cum_injury_total']]
#     df_avg_age['name'] = 'avg cum_injury_total'

#     df_player_vs_avg_age = pd.concat([df_player2, df_avg_age])

#     chart4 = alt.Chart(df_player_vs_avg_age).mark_line().encode(x=alt.X('age:Q'), y='cum_injury_total:Q', color='name'). \
#         properties(width=800, height=300)
#     st.altair_chart(chart4, use_container_width=False)

# # Plotting Chart 5 Compare Player Injury History vs. the Average Player's Injuries
#     st.subheader("Compare Player Injury History vs. the Average Player's Injuries")
#     st.write('* Player ages are updated with the latest data we have *')

#     player5 = st.selectbox("Name (type or choose):",sorted_unique_player)
    
#     df_picked_player = dataset[dataset['name'] == player5][['cum_week', 'name', 'Min', 'cum_injury_total']]
#     df_picked_player['cum_Min'] = df_picked_player['Min'].cumsum()

#     cum_Min_max = df_picked_player['cum_Min'].max()

#     df_avg_min = dataset[['cum_week', 'name', 'Min', 'cum_injury_total']]
#     df_avg_min['cum_Min'] = df_avg_min.groupby(by=['name'])['Min'].cumsum()
#     df_avg_min = df_avg_min.groupby('cum_week').mean().reset_index()
#     df_avg_min['name'] = 'avg of all players'

#     df_avg_min = df_avg_min[df_avg_min['cum_Min'] <= cum_Min_max]

#     df_picked_player.drop_duplicates(inplace=True)
#     df_avg_min.drop_duplicates(inplace=True)

#     df_player_vs_avg_min = pd.concat([df_picked_player, df_avg_min])

#     chart5 = alt.Chart(df_player_vs_avg_min).mark_line().encode(x=alt.X('cum_Min:Q'), y='cum_injury_total:Q', color='name'). \
#         properties(width=800, height=300)
#     st.altair_chart(chart5, use_container_width=False)
# 
elif section == "Interactive Injury Prediction Tool (BETA)":

     st.header("Interactive Injury Prediction Tool (BETA)")

else:
    st.header("Conclusions and Future Work")
