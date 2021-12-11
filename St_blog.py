import streamlit as st
from PIL import Image
import pandas as pd

img = Image.open("images/crowd-gc46d97eb2_1920.jpg")

st.image(img)
st.title("Sooner or later?  Walkthrough to predict when an elite soccer player will get injured.")

st.write("Sravan Rajupalem") 
st.write("Renzo Maldonado")
st.write("Victor Ruiz is in Orlando")

section = st.sidebar.selectbox("Sections", ("Scraping the Web for Data", "Data Manipulation", "Feature Engineering", 
    "Visual Exploration of Data", "Model Building"))

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
    
elif section == "Data Manipulation":
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
#     df_final = pd.DataFrame(columns=['Variable', 'Description'])
#     df_final['Variable'] = ['name', 'FBRefID', 'date', 'agg_week', 'agg_year', 'Injury', 'injury_week', 'injury_year', 'Min', 'Gls',
#         'Ast', 'PK', 'Pkatt', 'Sh', 'SoT', 'CrdY', 'CrdR', 'Touches', 'Press', 'Tkl', 'Int', 'Blocks', 'xG', 'npxG', 'xA', 'SCA', 
#         'GCA', 'Cmp', 'Att', 'Prog', 'Carries', 'Prog.1', 'Succ', 'Att.1', 'Fls', 'Fld', 'Off', 'Crs', 'TklW', 'OG', 'PKwon', 'Pkcon',
#         'Won', 'Loss', 'Draw', 'release_week', 'was_match', 'Height'
# Weight
# Birth
# cum_week
# defender
# attacker
# midfielder
# goalkeeper
# age
# right_foot
# left_foot
# injury_count
# cum_injury

# ']