import streamlit as st
from PIL import Image
import pandas as pd

st.markdown("![Alt Text](https://cdn.pixabay.com/photo/2016/03/27/19/03/crowd-1283691_1280.jpg)")
st.title("Sooner or later?  Walkthrough to predict when an elite soccer player will get injured.")

st.write("Sravan Rajupalem") 
st.write("Renzo Maldonado")
st.write("Victor Ruiz is in Orlando")

section = st.sidebar.selectbox("Sections", ("Scraping the Web for Data", "Data Manipulation & Feature Engineering", 
    "Visual Exploration of Data", "Model Building", "Injury Prediction Tool"))

st.write("""For quite a while, 'Sports Analytics' has been the buzz-word in the world of Data Science. Magically using complex 
    algorithms, machine learning models and neural networks to predict sports results and players' performance attract the interest 
    of people for different reasons. Soccer is probably one of the most unpredictable sports out there. In the hope of aiding soccer 
    managers' decisions, we decided to apply Data Science tools to predict how likely a player was to have an injury within a 
    certain time frame.""")

if section == "Scraping the Web for Data":
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
    st.write("From FBRef.com we first scraped information from the big 5 European leagues. With that base, we again scraped the \
        website for all the seasons. Then we scraped the player information from each of those seasons.  This operation yielded \
        81,256 player records. Finally we again scraped all players' urls to get all the matches that each player had participated \
        in. After going through these 5 iterations of scraping from FBRef.com, we obtained a list of 118,283 match logs. With this \
        list we again scraped the website by batches to obtain a final match logs data set, that after some NaN cleaning, data type \
        conversion  and dropping unwanted columns, ended up with a DataFrame named consolidated_df that had 3,048,121 rows with 47 \
        columns.")
    img5 = Image.open("images/image5.png")
    st.image(img5)
    st.write("Then we again scraped fbref.com on a per country basis to obtain each player's statistical information. This yielded \
        the following DataFrames:")
    table = pd.DataFrame(columns=['Country', 'DataFrame Name', 'Rows', 'Columns'])
    table['Country'] = ['England', 'Italy', 'Spain', 'France', 'Germany']
    table['DataFrame Name'] = ['player_data_df_england', 'player_data_df_italy', 'player_data_df_spain', 'player_data_df_france', \
        'player_data_df_germany']
    table['Rows'] = [6626,8255,7274,7354,6318]
    table['Columns'] = [15,15,15,15,15]
    table
    st.write("The transfermarkt.com website was our source for detailed data about each player's injury history. A similar \
        scraping process to the one used with the fbref.com website was applied here.  First scraping the league urls, then using \
        these league urls to scrape the team urls and then these team urls to find the player urls. Finally we used the player urls \
        to scrape the injury information for each player. This yielded a DataFrame with shape (55,216, 8) named player_injuries_df.")
    player_injuries_df = pd.read_csv('player_injuries_df.csv')
    player_injuries_df
    st.write("Additional information about the players' profile was scraped from transfermarkt.com by using the player urls.  This \
        process was done in batches of 4,000 records and it yielded the following DataFrames:")
    table2 = pd.DataFrame(columns=['DataFrame Name', 'Shape'])
    table2['DataFrame Name'] = ['player_profile_df', 'player_profile_df_2', 'player_profile_df_3']
    table2['Shape'] = ['(4000, 41)', '(4000, 41)', '(4000, 41)']
    table2
    st.write("The complete scraping process to get the data was done using the [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) \
        Python library.")
    
elif section == "Data Manipulation & Feature Engineering":
    st.header("Merging, Cleaning and Manipulating the Data")
    img6 = Image.open("images/image6.jpg")
    st.image(img6)
    st.write("The first step was to merge our player_profile_df_1, player_profile_df_2, and player_profile_df_3 tables using simple \
        concatenation and dropping duplicates which resulted in a DataFrame named tm_profile_df of 12,902 rows and 41 columns.")
    st.write("Then we created the player_info_df by concatenating the previously created DataFrames player_info_england, \
        player_info_italy, player_info_spain, player_info_france, and player_info_germany.  The resulting player_info_df had a \
        shape of 35,827 rows by 15 columns.")
    st.write("Next we had to use a handy mapping DataFrame called fbref_to_tm_mapping to link the websites' different ID's \
        (FBRefID and TMID) which we downloaded from [Jason Zickovic](https://github.com/JaseZiv/worldfootballR_data/tree/master/raw-data/fbref-tm-player-mapping/output) (kudos!). Via string conversions and splitting we extracted the \
        FBRefID's and TMID's and included them as columns.")
    st.write("We then merged on the intersection of player_injuries_df and fbref_to_tm_df on columns TMId and TMID respectively.")
    st.write("Several NaN cleaning, filling, dummy variable creation and replacement operations had to be done in order to get \
        new_player_df, our final DataFrame, which had a shape of 1,680,385 rows and 62 columns. The features of the new_player_df:")
    df_final = pd.DataFrame(columns=['Variable', 'Description'])
    df_final['Variable'] = ['name', 'FBRefID', 'date', 'agg_week', 'agg_year', 'Injury', 'injury_week', 'injury_year', 'Min', 'Gls',
        'Ast', 'PK', 'Pkatt', 'Sh', 'SoT', 'CrdY', 'CrdR', 'Touches', 'Press', 'Tkl', 'Int', 'Blocks', 'xG', 'npxG', 'xA', 'SCA', 
        'GCA', 'Cmp', 'Att', 'Prog', 'Carries', 'Prog.1', 'Succ', 'Att.1', 'Fls', 'Fld', 'Off', 'Crs', 'TklW', 'OG', 'PKwon', 'Pkcon',
        'Won', 'Loss', 'Draw', 'release_week', 'was_match', 'Height', 'Weight', 'Birth', 'cum_week', 'defender', 'attacker', \
        'midfielder', 'goalkeeper', 'age', 'right_foot', 'left_foot', 'injury_count', 'cum_injury']
    df_final['Description'] = ['Name of soccer player', 'FBRef Id', 'Date of the occurrence: game, injury or both', \
        'Week of the occurrence: game, injury or both', 'Year of the occurrence: game, injury or both', 'Type of injury', \
        'Week when injury occurred', 'Year when injury occurred', 'Minutes played', 'Goals scored or allowed', 'Completed assists', \
        'Penalty kicks made', 'Penalty kicks attempted', 'Shots (not including penalty kicks)', \
        'Shots on target (not including penalty kicks)', 'Yellow cards', 'Red cards', 'Touches in attacking penalty area', \
        'Passess made while under pressure of opponent', 'Number of players tackled', 'Interceptions', \
        'Number of times blocking the ball by standing on its path', 'Expected goals', 'Non-penalty expected goals', \
        'Expected assists previous to a goal', 'Two offensive actions previous to a shot', \
        'Two offensive actions previous to a goal', 'Passess completed', 'Passess attempted', \
        'Passess that move the ball at least 10 yards toward opponent goal', \
        'Number of times player controlled the ball with his feet', \
        "Carries that move the ball toward opponent's goal at least 5 yards", 'Dribbles completed successfully', \
        'Dribbles attempted', 'Fouls committed', 'Fouls drawn', 'Offsides', 'Crosses', \
        'Tackles were possession of the ball was won', 'Own goals', 'Penalty kicks won', 'Penalty kicks conceded', 'Game won', \
        'Game lost', 'Game draw', 'Week when player was released from injury', \
        'If there was a match during that week, variable = 1', 'Player height', 'Player weight', 'Player date of birth', \
        'Cumulative week', 'Player is a defender', 'Player is an attacker', 'Player is a midfielder', 'Player is a goalkeeper', \
        'Player age', 'Player is right-footed', 'Player is left-footed', 'Player number of injuries', 'Player cumulative injuries']
    df_final

    st.write("After all the data manipulation and feature engineering the head of our data frame looks like so:")
    dataset_for_model_final_head = pd.read_csv('dataframes_blog/dataset_for_model_final_head.csv')
    st.code(dataset_for_model_final_head.head())

elif section == "Visual Exploration of Data":
    st.header('Visual Exploration of Data')
    st.markdown("![Alt Text](https://cdn.pixabay.com/photo/2014/10/14/20/24/football-488714_1280.jpg)")
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



    

elif section == "Model Building":
    st.header("Model Building")

else:
    st.header('Injury Prediction Tool')

