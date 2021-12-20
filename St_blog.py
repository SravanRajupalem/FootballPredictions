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
import plotly.express as px
import plotly.graph_objects as go
# from streamlit_player import st_player

newlogo = Image.open("images/new_logo.png")

st.sidebar.image(newlogo)
section = st.sidebar.selectbox("Sections", ("Introduction", "Data Scraping", "Data Manipulation", 
    "Data Exploration", "Data Modeling", "Injury Prediction", "Exploration Tool (BETA)", 
    "Injury Prediction Tool (BETA)", "Conclusion", "Statement of Work"))

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #32AEB3;">
  <a class="navbar-brand" href="https://github.com/SravanRajupalem/FootballPredictions" target="_blank">GitHub</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav"> 
      <li class="nav-item active">
        <a class="nav-link" href="https://www.youtube.com/channel/UCawuViij7J33V5zscJH2G8A/videos" target="_blank">YouTube</a>
      </li>
      <li class="nav-item active">
        <a class="nav-link" href="https://fbref.com" target="_blank">FBRef</a>
      </li>
      <li class="nav-item active">
        <a class="nav-link" href="https://transfermarkt.com" target="_blank">TransferMarkt</a>
      </li>
      <li class="nav-item active">
        <a class="nav-link" href="mailto:providemus2021@gmail.com" target="_blank">Contact Us</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)


if section == "Introduction":
    # imgstadium = Image.open("images/stadium1.png")
    # st.image(imgstadium)
    st.video('https://youtu.be/bnq4hXZoCt0')

    st.title("Anticipate Sooner")
         
    st.title("Walkthrough to predict when an elite soccer player will get injured.")

    st.write("Sravan Rajupalem (sravanr@umich.edu)") 
    st.write("Renzo Maldonado (renzom@umich.edu)")
    st.write("Victor Ruiz (dsvictor@umich.edu)")
    st.markdown("***")


    st.write("<p style='text-align: justify; font-size: 16px'>For quite a while, 'Sports Analytics' has been the buzzword \
        in the world of Data Science. Magically using complex\
        algorithms, and machine learning models to predict sports results and players' performance attract the interest \
        of people for different reasons. Soccer is probably one of the most unpredictable sports out there. In the hope of aiding soccer \
        managers' decisions, we decided to apply Data Science tools to predict how likely a player is to have an injury within a \
        certain time frame.</h1>", unsafe_allow_html=True)

    st.write("<p style='text-align: justify; font-size: 16px'>Presenting Providemus, a tool to predict when a player will get injured. \
        By using data from the most reliable international soccer sources, our data scientists have been able to train machine learning \
        models to predict with considerable accuracy when a player will get injured. Five horizons of prediction have been considered: \
        a week, a month, a quarter, a semester, or a year. This system is meant to be used as a complementary tool for \
        soccer managers in their decisions to play or rest their players.</h1>", unsafe_allow_html=True)

###############################################################################################################################

elif section == "Data Scraping":

    imgcorner = Image.open("images/corner.jpg")
    st.image(imgcorner)
    
    st.header('Data Scraping')
    st.write("<p style='text-align: justify; font-size: 16px'>We hunted the web to get the most information we could about soccer \
        players and matches. After scanning several options and evaluating the completeness of their data, we decided to work with \
        fbref.com and transfermarkt.com.</h1>", unsafe_allow_html=True)
    img = Image.open("images/image1.png")
    img2 = Image.open("images/image2.png")
    img3 = Image.open("images/image3.png")
    st.image(img)
    st.image(img2)
    st.image(img3)
    st.write("<p style='text-align: justify; font-size: 16px'>The first major decision was to only get information from the five most competitive soccer leagues in \
        the world: the world: Premier League (England), La Liga (Spain), Bundesliga (Germany), Ligue 1 (France), and the Serie A (Italy). \
        We selected the top 5 European leagues because we believed that these leagues would have more data of players and matches.</h1>", unsafe_allow_html=True)
    img4 = Image.open("images/image4.png")
    st.image(img4)
    st.write("<p style='text-align: justify; font-size: 16px'>From FBRef we first scraped the URLs from the big 5 European leagues. With that base, we again scraped the website \
        for all the seasons for each league. Then we scraped the players' URLs from each of all available seasons of the top 5. \
        This operation yielded a list of 78,959 unique records. Those embedded URLs contained an identifier (FBRefID) for each of the \
        19,572 players found. Moreover, since we intended to scrape complementary players' data from the TransferMarkt \
        website, we decided to only pull data for the players whose information was available on both sites.</h1>", unsafe_allow_html=True)   
    st.write("Next, we had to use a handy mapping dataset called fbref_to_tm_mapping that links the websites' unique identifiers \
        FBRefID and TMID (TransferMarkt ID), which we downloaded from [Jason Zickovic](https://github.com/JaseZiv/worldfootballR_data/tree/master/raw-data/fbref-tm-player-mapping/output) \
        (kudos!).")
    st.write("<p style='text-align: justify; font-size: 16px'>Via string conversions and splitting we extracted the FBRefID's from the generated list and decided to only scrape \
        the match logs for those players from the FBRef site.</h1>", unsafe_allow_html=True)   
    
    img5 = Image.open("images/image5.png")
    st.image(img5)
    st.write("<p style='text-align: justify; font-size: 16px'>This effort helped us reduce a significant amount of memory usage when performing the data scrapping given that only 5,192 \
        players had attainable data from both sites. We executed another pull and obtained a list of 51,196 complete \
        match logs URLs of all the consolidated players.</h1>", unsafe_allow_html=True)
    img5a = Image.open("images/image5a.PNG")
    st.image(img5a)
    st.write("<p style='text-align: justify; font-size: 16px'>This is where the real data scraping of the players' match logs began. The extraction of all players' matches required high \
        computation; thus, our team divided the data extraction into multiple batches, where we extracted the match logs from each batch individually. \
        In the end, all these datasets were concatenated into a final dataframe we named consolidated_df_final.</h1>", unsafe_allow_html=True)
    img5b = Image.open("images/image5b.PNG")
    st.image(img5b)
    st.write("<p style='text-align: justify; font-size: 16px'>As we started building our main dataset, we began to understand more of the potential features that were going to be included in \
        our Machine Learning models. We quickly realized that players' profile data was critical in order to generate predictions. Attributes such as \
        the age of a player must be relevant to our predictions.</h1>", unsafe_allow_html=True)
    img5c = Image.open("images/image5c.gif")
    st.image(img5c)
    st.write("<p style='text-align: justify; font-size: 16px'>The older you get, the most likely you are to get injured... Right? Before we spun the wheels, we had to push down on the brakes and head back to \
        the fbref.com website to harvest more data. This process was similar, but in this case, we scraped information on a per-country basis to \
        obtain each player's profile information. This yielded the following DataFrames:</h1>", unsafe_allow_html=True)
    table = pd.DataFrame(columns=['Country', 'DataFrame Name', 'Rows', 'Columns'])
    table['Country'] = ['England', 'Italy', 'Spain', 'France', 'Germany']
    table['DataFrame Name'] = ['player_data_df_england', 'player_data_df_italy', 'player_data_df_spain', 'player_data_df_france', \
        'player_data_df_germany']
    table['Rows'] = [6626,8255,7274,7354,6318]
    table['Columns'] = [15,15,15,15,15]
    table
    st.write("<p style='text-align: justify; font-size: 16px'>Once all tables were completed, they were combined into a single dataframe of all players' profiles, where we ended up with \
        10,720 players. However, we only used 5,192 players since those had data available from both sources. Here is the new players_info_df:</h1>", unsafe_allow_html=True)
    img5d = Image.open("images/image5d.png")
    st.image(img5d)
    st.write("")
    st.write("<p style='text-align: justify; font-size: 16px'>The transfermarkt.com website was our source for detailed data about each player's injury history. \
        A similar scraping process to the one used with the fbref.com website was applied here. First, we scraped the league URLs, then we used these league URLs \
        to scrape the team URLs, and then those team URLs were utilized to find the player URLs. Finally, we employed the player URLs to scrape the injury information\
        for each player. This yielded a dataset of 55,216 rows and 8 columns named player_injuries_df.</h1>", unsafe_allow_html=True)
    player_injuries_df = pd.read_csv('player_injuries_df.csv')
    player_injuries_df
    st.write("<p style='text-align: justify; font-size: 16px'>Additional information about the players' profiles was scraped from transfermarkt.com by using the player URLs.  This \
        process was accomplished employing 3 batches of 4,000 records and it yielded the following DataFrames:</h1>", unsafe_allow_html=True)
    table2 = pd.DataFrame(columns=['DataFrame Name', 'Shape'])
    table2['DataFrame Name'] = ['player_profile_df_1', 'player_profile_df_2', 'player_profile_df_3']
    table2['Shape'] = ['(4000, 41)', '(4000, 41)', '(4000, 41)']
    table2
    st.write("<p style='text-align: justify; font-size: 16px'>This dataset contained additional information that the FBRef site did not provide. Here we found new \
        attributes such as the date a player joined a club, the date they retired, and other features we believed could be useful. Were all these features \
        actually used in our models? Please stay tuned...</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align: justify; font-size: 16px'>Here is the new tm_profile_df dataset after the concatenation:</h1>", unsafe_allow_html=True)
    img5f = Image.open("images/image5f.PNG")
    st.image(img5f)
    st.write("")
    st.write("The complete scraping process to harvest the data was done using the [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) \
        Python library")

###############################################################################################################################

elif section == "Data Manipulation":

    img6 = Image.open("images/image6.jpg")
    st.image(img6)
   
    st.header("Data Manipulation")

    st.write("<p style='text-align: justify; font-size: 16px'>At this point we inspected, cleaned, transformed, and merged our datasets with the ultimate goal of \
        producing a final dataset to train our machine learning models. We achieved this by merging on the \
        intersection of all dataframes using the fbref_to_tm_df reference table on columns FBRefID (FBRef unique IDs) and TMID (TransferMarkt unique IDs) respectively. This phase \
        of the project required unbiased analysis and the evaluation of how each attribute could contribute to our models as well as trial and error to experiment \
        with various methods to find the most successful features. We also needed to avoid adding redundant variables as this could have reduced the \
        generalization capability of the model and decreased the overall accuracy. Attributes such as a player's number of minutes played could imply that the more a player \
        plays, the more likely a player is to get injured. Thus, we concluded that this feature had to be included. On the other hand, we first \
        believed that weight could have also been a key feature to keep. However, most soccer players have to go through rigorous training and \
        stay in shape; thus, players' weights did not contribute much to our models. Additionally, our data also gave us room to reengineer some \
        features. Moreover, we created additional features from our existing dataset. Who is more likely to get injured? A goalkeeper or an attacker? \
        We were not sure. Again, at this stage, we were just learning and discovering trends \
        from our data. Furthermore, we created dummy variables to distinguish the positions of the players. Did the position of the player contribute \
        to our model? We will see!</h1>", unsafe_allow_html=True)
    img5e = Image.open("images/image5e.jpg")
    st.image(img5e, use_column_width ="always") 
    st.write("")
    st.write("<p style='text-align: justify; font-size: 16px'>Before defining our features, we first merged all of our datasets: consolidated_df_final (FBRef match logs), players_info_df \
        (FBRef profiles), and player_injuries_df (TransferMarkt injuries). We named this new dataframe as player_injuries_profile_final, \
        which yielded a dataframe of 159,362 rows and 75 columns. However, this dataset changed several times since many steps were taken as we were cleaning and defining \
        all features. Removing duplicates, dropping NaNs, updating column types, and many other basic operations were applied. Most importantly, we \
        aggregated all columns at the week level. In other words, our final dataset contained all players' profile data, match logs, and injuries at the \
        week level. For example, a football player played 2 entire games within a week; then the footballer  played a total of 180 minutes. The same \
        concept arose when a player scored in multiple games within a week; if a player scored a hattrick on Tuesday and a brace on Sunday, then a \
        single instance(row) of the data showed that this player had 5 goals. This step aggregated all column values with the groupby function and the sum() \
        operator. This was a critical step for our time series models. Likewise, we added the weeks when players did not play and filled those with 0s. \
        That is to say, if a player didn't play a certain week, then we added a row and populated all date columns accordingly and the remaining columns \
        were loaded with 0s. Additionally, we created new columns to include the week and year a player gets injured as well as the week the player is released.</h1>", unsafe_allow_html=True)
    st.write("")
    st.write("<p style='text-align: justify; font-size: 16px'>This is how the dataset looked before we aggregated the dates:</h1>", unsafe_allow_html=True) 
    img14 = Image.open("images/image14.PNG")
    st.image(img14) 
    st.write("<p style='text-align: justify; font-size: 16px'>As we explored the dataset, new features were developed. To name a few more, we constructed new columns to highlight \
        when a player's team wins, loses, has a draw. When we thought of this, it was also determined to incorporate another feature to state when a player starts \
        the game from the beginning.</h1>", unsafe_allow_html=True) 
    img13 = Image.open("images/image13.PNG")
    st.image(img13) 
    st.write("<p style='text-align: justify; font-size: 16px'>We believed competitions or tournaments where players participated could influence our model, especially when players are on international duty \
        to play for major tournaments such as the world qualifiers. If a football player gets injured due to international duty, this creates friction between the club and \
        the player's national team. We are interested in understanding how these variables affect players' performance and how it makes them prone to injury. \
        Several questions arose. Are players more prone to get injured if they perform better? \
        Are players more likely to get hurt during a world cup qualifying game? Does the venue \
        of a game have an influence on a player's probability of injury?</h1>", unsafe_allow_html=True)
    img15 = Image.open("images/image15.PNG")
    st.image(img15)
    
    st.write("<p style='text-align: justify; font-size: 16px'>Consequently, we created dummy variables to come up with new features for all competitions available and the venue.</h1>", unsafe_allow_html=True)
    img16 = Image.open("images/image16.PNG")
    st.image(img16)
    
    img17 = Image.open("images/image17.jpg")
    st.image(img17)
    st.write("<p style='text-align: justify; font-size: 16px'>After all the data manipulation process and feature engineering, we produced a new dataset we named complete_final_df_all with. It consisted of 1,910,255 rows and 169 \
        columns for a total of 4,588 players. We were then ready to start building our data models, but first, let's take a look at the dataset. Here we \
        want to show you a subset of one of Cristiano Ronaldo's best seasons during the time he lead Real Madrid to win 'La Decima' where he broke an all-time \
        record and scored 17 goals in a single season for the Champions League.</h1>", unsafe_allow_html=True) 
    st.write("<p style='text-align: justify; font-size: 16px'>Feel free to scroll up, down, left, and right", unsafe_allow_html=True) 
    cr7_df = pd.read_csv('dataframes_blog/df_cristiano.csv')
    cr7_df
    
    st.write("<p style='text-align: justify; font-size: 16px'>We continued creating features in the next sections...</h1>", unsafe_allow_html=True)

###############################################################################################################################

elif section == "Data Exploration":

    img7 = Image.open("images/ball.png")
    st.image(img7, width = 700)
    
    st.header('Data Exploration')
    st.write("<p style='text-align: justify; font-size: 16px'>Data exploration helps us understand the relationships between the dependent and the independent \
        variables. The dataset contained 2 possible classes in the target variable: 0 if a player is not injured and 1 if a player is \
        injured. The target value was studied at different time windows to see how probable it was that the player would get injured in \
        the next week, quarter, semester, or the next year. These classes had the following proportions:</h1>", unsafe_allow_html=True)
    img7 = Image.open("images/image7.png")
    st.image(img7)
    st.write("<p style='text-align: justify; font-size: 16px'>We can see that the dominant class is 0: when players are not injured, which makes sense because we don't expect players \
        to be injured more time than they are not injured. So our data is unbalanced which had to be taken into account when we \
        modeling.</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align: justify; font-size: 16px'>We did some additional explorations to see if the data made 'sense'.  We wanted to observe the relationship between minutes \
        played and the age of players and the positions they played in.</h1>", unsafe_allow_html=True)
    img8 = Image.open("images/image8.png")
    st.image(img8)
    st.write("<p style='text-align: justify; font-size: 16px'>In this case, it's interesting to see that there seems to be an 'optimal' age where players tend to play more minutes. \
        It looks like players between 20 and 34 years old play more minutes. This is unexpected, as we would have thought younger \
        players would play more minutes, but on second thought, it makes sense due to their career development.</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align: justify; font-size: 16px'>We also plotted minutes played (Min) vs the accumulated number of injuries per player (cum_injury_total).</h1>", unsafe_allow_html=True)
    img9 = Image.open("images/image9.png")
    st.image(img9)
    st.write("<p style='text-align: justify; font-size: 16px'>We found a logical pattern, players with less accumulated injuries tend to play more minutes.</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align: justify; font-size: 16px'>Additionally, we plotted the players' weight (Weight) vs the accumulated number of injuries per player (cum_injury_total).</h1>", unsafe_allow_html=True)
    img10 = Image.open("images/image10.png")
    st.image(img10)
    st.write("<p style='text-align: justify; font-size: 16px'>Here there seems to be a concentration of players between 65 kilos and 85 kilos that gets more injuries. This \
        is probably due to the fact that we are looking at proffesional players in the most competitive leagues.  Most of them \
        weigh in that range.</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align: justify; font-size: 16px'>We decided to plot a correlation matrix (heatmap style) to look at our whole universe of variables.</h1>", unsafe_allow_html=True)
    img11 = Image.open("images/image11.png")
    st.image(img11)
    st.write("<p style='text-align: justify; font-size: 16px'>As we can see, com_injury_total seems to have a higher positive correlation with variables like Weight and Age.</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align: justify; font-size: 16px'>To get idea of how variables are correlated we 'zoomed-in' to just 5 variables like: 'Height', 'Weight', 'age', \
        'cum_injury_total', 'weeks_since_injury', and 'Min_cum'.</h1>", unsafe_allow_html=True)
    img12 = Image.open("images/image12.png")
    st.image(img12)
    st.write("<p style='text-align: justify; font-size: 16px'>Here we can analyze if there are any correlations between variables.  Logically, height is positively correlated with \
        weight. Being that we are analyzing active athletes, there doesn't seem to be any correlation between weight and age. We also \
        see a positive correlation between age and the cum_injury_total.</h1>", unsafe_allow_html=True)
    st.header("Deeper Data Exploration")
    st.write("<p style='text-align: justify; font-size: 16px'>The following set of tools has been developed to evaluate and understand how players' injuries evolve over time. It is reasonable to \
        assume that as players age, they are more likely to become injured. Of course, there are some players that do not get injured as much as others \
		while some get injured more frequently. The following tools are intended to help us understand some of the differences \
        between these players.</h1>", unsafe_allow_html=True) 
    st.subheader("Compare Players' Injury History")
    st.write("<p style='text-align: justify; font-size: 16px'>First, we want to individually compare players' injury history. Here, you are able to select up to 3 players. This line chart exhibits a clear comparison of \
        how players start accumulating injures throughout time, where time is displayed in cumulative weeks. For this example, we are showing a model comparison on \
        3 players with similar attributes. Football fans love to compare Ronaldo and Messi. They both have had an amazing career, have scored many goals, and have broken \
        all the records, and it seems that they will continue to do so for the next few years.  However, Leo has accumulated more injuries than CR7 even though Leo has \
        played fewer games. We are including Leo and Cristiano for our comparison, and also incorporate Robert Lewandoski since he has become a 'scoring machine' the past \
        years. It seems that Lewandoski is projected to accumulate fewer injuries than Leo. Why did Lionel Messi accumulate more injuries than Cristiano? Lionel Messi plays \
        the South American qualifiers and Copa America while Cristiano and Lewandoski play the European qualifiers and the Eurocup. Is it because of the competitions? \
        One thing to mention is that Messi is a midfielder while Cristiano and Lewandoski are strikers. May the position of the player influence the likeliness of a player \
        getting injured? We mentioned earlier that goalkeepers may not be as exposed as other positions. However, they do get injured. The following visualization will \
        help us answer this.</h1>", unsafe_allow_html=True)
    img18 = Image.open("images/image18.PNG")
    st.image(img18)
    st.subheader("Compare Cumulative Injury History According to Position")
    st.write("<p style='text-align: justify; font-size: 16px'>This tool allows you to compare the variation of injuries based on the player's position. We have previously defined 4 different positions for each player, \
        in this tool, you can compare all players' positions at the same time. When we only select the goalkeeper position, we can immediately notice that they \
        are less likely to get injured compared to other positions. Now, we can strongly say that goalkeepers have a much lower chance of getting injured. Furthermore, \
        you may think that attackers are the ones with a higher risk of getting injured; however, this graph verifies that defenders are the ones that accumulate the most injuries.</h1>", unsafe_allow_html=True)
    img19 = Image.open("images/image19.PNG")
    st.image(img19)
    st.subheader("Compare Player Injury History vs. the Average Injuries in the Position He Plays")
    st.write("<p style='text-align: justify; font-size: 16px'>The next visualization helps us compare a player's injuries through his entire career against the average injuries of players in the same position of \
        the player you select. It is crucial to understand how this player is doing compared to other players in his position. Here you can visualize where a \
        player stands with other similar players. In this case, Gareth Bale has been appointed since he is a phenomenal player who unfortunately has had had many injuries during his football career. As shown, he has \
        accumulated more injuries than the cumulative average of all the attackers from our dataset. Thus, this is a player that may continue to get injured until the \
        end of his career.</h1>", unsafe_allow_html=True)
    st.subheader("Compare Player Injury History vs. the Average Injuries for His Age")
    img20 = Image.open("images/image20.PNG")
    st.image(img20)
    st.write("<p style='text-align: justify; font-size: 16px'>Here, we wanted to take a very similar approach by comparing a player's injury history against the average injuries for players of the same age as the selected player. Again, \
        we used Gareth Bale as an example, and the same trend occurs where he's above the average cumulative total injuries of players his age. Bale's \
        injuries have continued to increase at a steady phase during the years. It seems that he has not been able to have a full season without injuries. He is \
        just one of those players who keeps suffering from injury setbacks.</h1>", unsafe_allow_html=True)
    img21 = Image.open("images/image21.PNG")
    st.image(img21)
    st.subheader("Compare Player Injury History vs. the Average Player's Injuries")
    st.write("<p style='text-align: justify; font-size: 16px'>Last, this is a comparison of a single player's injuries history against the average injuries of all players. The x-axis represents the cumulative minutes \
        of all games played, and on the y-axis, the graph displays the cumulative injuries through time. Again, we chose Robert Lewandoski. As \
		shown on this graph, as he started to accumulate minutes at the beginning of his career, he wouldn't get as many injuries as the average football player. Once he \
		reaches over 40,000 minutes, he overtakes this average and starts to accumulate more injuries than the average player. It's also important to state that \
        after a certain amount of accumulated minutes, injuries tend to decrease, this is because as players get older they tend to play \
        tend to play less.</h1>", unsafe_allow_html=True)
    img22 = Image.open("images/image22.PNG")
    st.image(img22)

###############################################################################################################################

elif section == "Data Modeling":
    

    img8 = Image.open("images/footballfire.jpeg")
    
    st.image(img8, width = 700)
    
    st.header("Model Building")
    
    st.subheader("Target Variable Preparation")
    
    st.write("""<p style='text-align: justify; font-size: 16px'>Now we are into the model building phase of the project. The first thing we need to do is to specify the target variables. In this case, \
        we are looking at historic data of players to see when injuries occurred and try to use that information to anticipate when injuries are likely to happen in the future.\
            This was done by creating the target variable, whether a player got injured or not, using five different time periods:
            \n- One Week
            \n- One Month
            \n- One Quarter (3 months)
            \n- One Semester (6 months)
            \n- One Year (12 months)
            
            \nWe created the target variables in the dataset using the function below:
            """, unsafe_allow_html=True)
    
    st.code("""# Creating target column 'injured_in_one_week' and creating cumulative features
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
    """)
        
    st.write("""<p style='text-align: justify; font-size: 16px'>This function was used to determine an injury indicator shifted by the time horizons specified (note that 'FBRefID' was used as the unique player identifier).\
        For example, for the one-week horizon, if a player gets injured in week 60 in the data, \
        the "injured_in_one_week" column will show 1 in week 59. This can then be used as the target variable in the one-week horizon model using the range of the features specified in the previous section.\
            The reason this approach is taken is that this model is designed to be an anticipatory tool, hence there will be no value in predicting the exact instance when an injury will occur. Rather, we \
                would like managers to make informed decisions about their players by perhaps resting or focusing on rehab when injuries are predicted in the future, hence avoiding injuries before they happen.\
                    The same function was then applied to the count of injuries at each point in time for each player and also the cumulative injuries for each player at each point in time.
                """, unsafe_allow_html=True)
    
    st.subheader("Redefining Features")
    
    st.write("""<p style='text-align: justify; font-size: 16px'>Now that we have our target variables defined for each model we also need to make some adjustments to the features available in the data.\
        For example, the number of minutes played in matches should be an important indicator of potential injury concerns for a player. However, we would need a cumulative sum of this variable up until\
            each data point (i.e. number of cumulative minutes played until week 60). We will need this accumulation over the lifespan of the player's career. This was done via the function\
            below where any columns that required an accumulation over the player's career were recalculated. Other examples of accumulated features were the number of games won, lost, and drawn as well as the number\
                of games played in certain leagues (i.e. Champions League games of their career).
                """, unsafe_allow_html=True)
    
    st.code("""# Creating a function to add cumulative columns

def cummulative_sum(dataset, cum_column, original_column):
    dataset[cum_column] = dataset.groupby(['FBRefID', 'cum_sum'])[original_column].cumsum()
    return dataset
    
# Creating cummulative variables
cum_cols = ['Min', 'Gls', 'Ast', 'PK', 'PKatt', 'Sh', 'SoT', 'CrdY', 'CrdR', 'Touches', 'Press', 'Tkl', 'Int', 'Blocks', 'xG', 'npxG', 'xA', 
    'SCA', 'GCA', 'Cmp', 'Att', 'Prog', 'Carries', 'Prog.1', 'Succ', 'Att.1', 'Fls', 'Fld', 'Off', 'Crs', 'TklW', 'OG', 'PKwon', 'PKcon', 'Won', 
    'Loss', 'Draw', 'was_match']

for var in cum_cols:
    cummulative_sum(dataset, var+'_cum', var)
    """, language='python')

    st.write("""<p style='text-align: justify; font-size: 16px'>
        \n<p style='text-align: justify; font-size: 16px'>The first step before diving into the algorithms themselves was to drop any weeks in the dataset where the player had sustained an injury and was not playing. We are only interested in \
            capturing the week in which the injury occurred and any subsequent weeks in which the player was still recovering from the same injury is not useful for this model. We did however \
                explore an injury duration prediction model as well, which will predict the time that the player is expected to take once an injury has been sustained. We will briefly touch on this model later in this post.\
                    
                    \n<p style='text-align: justify; font-size: 16px'>Once the redundant injury weeks are removed from the dataset. We created the train and test datasets. Unlike normal machine learning models in which the train and test datasets \
                        can be randomly allocated across the dataset, this analysis requires a different approach as it is a time series analysis for each player. Hence, the \
                            number of weeks each player plays was determined, and the first 75% of the player's career in weeks was allocated to the training dataset and the remaining 25% was \
                                allocated to the test set (as shown below).
                """, unsafe_allow_html=True)
    
    st.code("""
    df_train = dataset[dataset['cum_week'] <= dataset["train_split"]].dropna()
    df_test = dataset[dataset['cum_week'] > dataset["train_split"]].dropna()
            """)
    
    st.write("""<p style='text-align: justify; font-size: 16px'>We selected a subset of the features in the data which we determined to be have some predictive power based on\
        trends we observed in the data exploration phase. They are outlined in the table below:
                """, unsafe_allow_html=True)
    
    df = pd.DataFrame(columns=['Features', 'Description'])
    
    extended_features = ['Height', 'Weight', 'defender', 'attacker', 'midfielder', 'goalkeeper', 'right_foot', 'age', 'cum_injury_total', 'weeks_since_last_injury', 'Min_cum', 'Gls_cum', 'Ast_cum', 'PK_cum', 'PKatt_cum',
                         'Sh_cum', 'SoT_cum', 'CrdY_cum', 'CrdR_cum', 'Touches_cum', 'Press_cum', 'Tkl_cum', 'Int_cum', 'Blocks_cum', 'xG_cum', 'npxG_cum', 'xA_cum', 'SCA_cum', 'GCA_cum', 'Cmp_cum',
                         'Att_cum', 'Prog_cum', 'Carries_cum', 'Prog.1_cum', 'Succ_cum', 'Att.1_cum', 'Fls_cum', 'Fld_cum', 'Off_cum', 'Crs_cum', 'TklW_cum', 'OG_cum', 'PKwon_cum','PKcon_cum', 'Serie A_cum',
                         'Premier League_cum', 'La Liga_cum', 'Ligue 1_cum', 'Bundesliga_cum', 'Champions Lg_cum', 'Europa Lg_cum', 'FIFA World Cup_cum', 'UEFA Nations League_cum', 'UEFA Euro_cum',
                         'Copa América_cum', 'Away_cum', 'Home_cum', 'Neutral_cum']
    
    descriptions = ['Height of Player in meters', 'Weight of Player in kilograms', 'Dummy variable of whether the player is a defender (1 if defender)', 
                    'Dummy variable of whether the player is a attacker (1 if attacker)', 'Dummy variable of whether the player is a midfielder (1 if midfielder)',
                    'Dummy variable of whether the player is a goalkeeper (1 if goalkeeper)', 'Dummy variable of whether the player is right footed (1 if right footed)',
                    'Age of Player at each data point', 'Total number of injuries till the datapoint for each player', 'Number of weeks since last injury', 
                    'Total number of Minutes played in matches till the datapoint for each player', 'Cumulative Goals scored or allowed', 'Cumulative Completed assists',
                    'Cumulative Penalty kicks made', 'Cumulative Penalty kicks attempted', 'Cumulative Shots (not including penalty kicks)',
                    'Cumulative Shots on target (not including penalty kicks)', 'Cumulative Yellow cards', 'Cumulative Red cards', 'Cumulative Touches in attacking penalty area',
                    'Cumulative Passes made while under pressure of opponent', 'Cumulative Number of players tackled', 'Cumulative Interceptions',
                    'Cumulative Number of times blocking the ball by standing on its path', 'Cumulative Expected goals', 'Cumulative Non-penalty expected goals',
                    'Cumulative Expected assists previous to a goal', 'Cumulative Two offensive actions previous to a shot',
                    'Cumulative Two offensive actions previous to a goal', 'Cumulative Passess completed', 'Cumulative Passes attempted',
                    'Cumulative Passess that move the ball at least 10 yards toward opponent goal',
                    'Cumulative Number of times player controlled the ball with his feet',
                    "Cumulative Carries that move the ball toward opponent's goal at least 5 yards", 'Cumulative Dribbles completed successfully',
                    'Cumulative Dribbles attempted', 'Cumulative Fouls committed', 'Cumulative Fouls drawn', 'Cumulative Offsides', 'Cumulative Crosses',
                    'Cumulative Tackles were possession of the ball was won', 'Cumulative Own goals', 'Cumulative Penalty kicks won', 'Cumulative Penalty kicks conceded',
                    'Cumulative games played in Serie A', 'Cumulative games played in Premier League', 'Cumulative games played in La Liga',
                    'Cumulative games played in Ligue 1', 'Cumulative games played in Bundesliga', 'Cumulative games played in Champions League',
                    'Cumulative games played in Europa League', 'Cumulative games played in FIFA World Cup', 'Cumulative games played in UEFA Nations League',
                    'Cumulative games played in UEFA Euro', 'Cumulative games played in Copa América', 'Cumulative games played Away', 'Cumulative games played Home',
                    'Cumulative games played in Neutral venues']
    
    df['Features'] = extended_features
    df['Description'] = descriptions
    
    df
    
    st.subheader('Classification Algorithms')
    
    st.write("""<p style='text-align: justify; font-size: 16px'>Now we are ready to proceed to build classification models for each of the time horizons. We will use the one-week model as an example\
        in this post but you can refer to the GitHub repository for the associated code for all the other time horizons. The general library used for machine learning in Python is Sci-kit Learn, however, \
            for this analysis, we have decided to implement the PyCaret library.""", unsafe_allow_html=True) 
    st.write("""[PyCaret](https://pycaret.org/) is open source low-code machine learning library with lots of cool functionality.""")
    
    st.write("""<p style='text-align: justify; font-size: 16px'>First, we set up the configuration of our model via the setup function in PyCaret. In this case, there are a couple of parameters we should tweak \
        such as adjusting for the imbalance in classes by setting 'fix_imbalance' parameter to 'True' and also the 'fold_strategy' to 'timeseries' to account for the time element in the dataset. We have also \
            allowed for the algorithm to select the most predictive features by letting 'feature_selection' equal 'True'. Due to computation and time constraints, the initial set of models was \
                to build using only 2 folds.
                """, unsafe_allow_html=True)

    st.code("""
injured_pred = 'injured_in_1_week'

extended_features = ['Height', 'Weight', 'defender', 'attacker', 'midfielder', 'goalkeeper', 'right_foot', 'age', 'cum_injury_total', 'weeks_since_last_injury', 'Min_cum', 'Gls_cum', 'Ast_cum', 'PK_cum', 'PKatt_cum',
 'Sh_cum', 'SoT_cum', 'CrdY_cum', 'CrdR_cum', 'Touches_cum', 'Press_cum', 'Tkl_cum', 'Int_cum', 'Blocks_cum', 'xG_cum', 'npxG_cum', 'xA_cum', 'SCA_cum', 'GCA_cum', 'Cmp_cum',
 'Att_cum', 'Prog_cum', 'Carries_cum', 'Prog.1_cum', 'Succ_cum', 'Att.1_cum', 'Fls_cum', 'Fld_cum', 'Off_cum', 'Crs_cum', 'TklW_cum', 'OG_cum', 'PKwon_cum','PKcon_cum', 'Serie A_cum',
 'Premier League_cum', 'La Liga_cum', 'Ligue 1_cum', 'Bundesliga_cum', 'Champions Lg_cum', 'Europa Lg_cum', 'FIFA World Cup_cum', 'UEFA Nations League_cum', 'UEFA Euro_cum',
 'Copa América_cum', 'Away_cum', 'Home_cum', 'Neutral_cum']
 
X_train = df_train[extended_features]
y_train = df_train[injured_pred]

X_test = df_test[extended_features]
y_test = df_test[injured_pred]
            
exp_clf = setup(dataset[extended_features + [injured_pred]], target=injured_pred, fix_imbalance=True, feature_selection=True, fold=2, fold_strategy='timeseries')      
            """)
    df = pd.DataFrame(columns=['Description', 'Value'])    

    description = ['session_id',	'Target',	'Target Type',	'Label Encoded',	'Original Data',	'Missing Values',	'Numeric Features',	'Categorical Features',	'Ordinal Features',	'High Cardinality Features',	'High Cardinality Method',	'Transformed Train Set',
               'Transformed Test Set',	'Shuffle Train-Test',	'Stratify Train-Test',	'Fold Generator',	'Fold Number',	'CPU Jobs',	'Use GPU',	'Log Experiment',	'Experiment Name',	'USI',	'Imputation Type',	'Iterative Imputation Iteration',	'Numeric Imputer',
               'Iterative Imputation Numeric Model',	'Categorical Imputer',	'Iterative Imputation Categorical Model',	'Unknown Categoricals Handling',	'Normalize',	'Normalize Method',	'Transformation',	'Transformation Method',	'PCA',	'PCA Method',	'PCA Components',	'Ignore Low Variance',
               'Combine Rare Levels',	'Rare Level Threshold',	'Numeric Binning',	'Remove Outliers',	'Outliers Threshold',	'Remove Multicollinearity',	'Multicollinearity Threshold',	'Remove Perfect Collinearity',	'Clustering',	'Clustering Iteration',	'Polynomial Features',	'Polynomial Degree',	
               'Trignometry Features',	'Polynomial Threshold',	'Group Features',	'Feature Selection',	'Feature Selection Method',	'Features Selection Threshold',	'Feature Interaction',	'Feature Ratio',	'Interaction Threshold',	'Fix Imbalance',	'Fix Imbalance Method']		

    value = ['4936',	'injured_in_1_week',	'Binary',	'0.0: 0, 1.0: 1',	'(1787492, 59)',	'TRUE',	'53',	'5',	'FALSE',	'FALSE',	'None',	'(1248469, 54)',
         '(534994, 54)',	'TRUE',	'FALSE',	'TimeSeriesSplit',	'2',	'-1',	'FALSE',	'FALSE',	'clf-default-name',	'00f0',	'simple',	'None',
         'mean',	'None',	'constant',	'None',	'least_frequent',	'FALSE',	'None',	'FALSE',	'None',	'FALSE',	'None',	'None',
         'FALSE',	'FALSE',	'None',	'FALSE',	'FALSE',	'None',	'FALSE',	'None',	'TRUE',	'FALSE',	'None',	'FALSE',
         'None',	'FALSE',	'None',	'FALSE',	'TRUE',	'classic',	'0.8',	'FALSE',	'FALSE',	'None',	'TRUE',	'SMOTE']

    df['Description'] = description
    df['Value'] = value

    df
    st.write("""<p style='text-align: justify; font-size: 16px'>The setup function outputs the above table with all the parameters flowing into the classification algorithms. Any of the parameters can be adjusted \
        using the set_config() method as we will do now to adjust the algorithms to use our train and test datasets rather than the ones compiled by PyCaret.
            """, unsafe_allow_html=True)
    
    st.code("""        
set_config('X_train', X_train)
set_config('X_test', X_test)
set_config('y_train', y_train)
set_config('y_test', y_test)
"""
)
    st.write("""<p style='text-align: justify; font-size: 16px'>Next we can run a range of classification algorithms all with one simple line of code. This produces an output with accuracy, AUC, recall, precision and F1 \
        across all the models for easy comparison.
            """, unsafe_allow_html=True)
    
    st.code("""best_model = compare_models()""")
    img9 = Image.open("images/results_1_week.png")
    st.image(img9, width = 700)
    
    st.markdown("""<p style='text-align: justify; font-size: 16px'> In this case, the model with the best overall accuracy was the _Dummy Classifier_, however, for this application in which we \
        have a highly imbalanced dataset, accuracy is not a good metric to evaluate the performance of the model. Instead, since we are predicting class labels and the positive class (i.e. whether \
        the player is injured) is more important with false positives and false negatives are equally costly for the prediction we will use the F1 Score as the primary metric for evaluation. \
        If we were more interested in the probability of injury rather than the pure classifictaion of injuries and non-injuries we may use the AUC metric instead (as shown in the diagram below).
            """, unsafe_allow_html=True)
    
    img10 = Image.open("images/Metric Selection Imbalanced.png")
    st.image(img10, width = 700)
    
    st.write("""<p style='text-align: justify; font-size: 16px'> Here is a summary of all the algorithms selected for each of the forecast horizons.
            """, unsafe_allow_html=True)
    
    df = pd.DataFrame(columns=['Horizon', 'Model', 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC', 'TT (Sec)'])
    df['Horizon'] = ['1 Week', '1 Month', '1 Quarter', '1 Semester', '1 Year']
    df['Model'] = ['Light Gradient Boosting Machine', 'Light Gradient Boosting Machine', 'Gradient Boosting Classifier', 'Ada Boost Classifier', 'Ada Boost Classifier']
    df['Accuracy'] = [0.9708, 0.946, 0.8344, 0.7407, 0.7482]
    df['AUC'] = [0.8228, 0.7051, 0.6682, 0.6246, 0.6062]
    df['Recall'] = [0.3684, 0.1647, 0.2907, 0.3896, 0.3527]
    df['Precision'] = [0.4547, 0.2725, 0.1259, 0.1153, 0.1217]
    df['F1'] = [0.407, 0.205, 0.1756, 0.1779, 0.181]
    df['Kappa'] = [0.3922, 0.1788, 0.0994, 0.0752, 0.0721]
    df['MCC'] = [0.3945, 0.185, 0.1102, 0.095, 0.0865]
    df['TT (Sec)'] = [23.68, 23.345, 424.82, 105.18, 117.04]
    
    df
    
    fig = px.scatter(df, x="Recall", y="Precision",
	         size="F1", color="Horizon",
                 hover_name="Model")
    
    st.plotly_chart(fig)
    
    st.write("""<p style='text-align: justify; font-size: 16px'> As shown in the above chart, injury prediction is a difficult task. The one-week model is by far the best performing with F1 score of <i>0.41</i>, \
        recall of <i>0.37</i> and precision of <i>0.45</i> using the <b>Light Gradient Boosting Machine</b>. This suggests that in any given week, if the model predicts that two players will get injured, one of the player actually will get injured. \
            This will be particularly useful for coaches for week-to-week management of the players. If they wanted to be cautious they could rest both players or if it is very important game \
                they may choose to take the risk and play both players. 
                
                \n<p style='text-align: justify; font-size: 16px'>The overall performance of the models reduces when we try to predict in longer time periods. Intuitively, this is because as the horizon gets longer \
                    there are more uncertainties and hence prediction is a more difficult task. There was some overfitting observed in the larger horizon models which was partially reduced by applying cross validation. \
                        In future work, this could be reduced by training with more data, removing features that are not predictive in the long-term, early stopping and ensembling.
                        
                        \n<p style='text-align: justify; font-size: 16px'> In the next section we will dive deeper into further tuning and evaluating the one-week model. 
            """, unsafe_allow_html=True)
    
    st.header('Model Evaluation')
    st.write("""<p style='text-align: justify; font-size: 16px'>After comparing all algorithms, we selected the model that performed the best in terms of F1 score, which was the \
        Light Gradient Boosting Machine for the one-week model. We then fit this model with 10 cross validation folds.""", unsafe_allow_html=True)
    
    st.code("""
    model = create_model(df.index[0], fold=10)
    save_model(model, 'model_1_week')
    """)
    
    img11 = Image.open("images/One Week Model Initial.png")
    st.image(img11, width = 500)
    
    st.write("""<p style='text-align: justify; font-size: 16px'>After fitting with 10 folds, we see that the average F1 score remains at 0.41. We then go ahead and perform hyperparameter tuning, \
        this is very simple when using PyCaret as it can be done with just one line of code: 
             """, unsafe_allow_html=True)
    
    st.code("""tuned_model = tune_model(model, optimize = 'F1')""")
    
    st.write("""<p style='text-align: justify; font-size: 16px'>Again, we have chosen to optimize the F1 score. Below are the results after tuning, it has improved the F1 score slight from <i>0.4084</i> \
        to <i>0.4110</i>.
             """, unsafe_allow_html=True)
    
    img12 = Image.open("images/One Week Model Tuned.png")
    st.image(img12, width = 400)
    st.code("""LGBMClassifier(bagging_fraction=0.8, bagging_freq=0, feature_fraction=0.9,
               learning_rate=0.15, min_child_samples=100, min_split_gain=0,
               num_leaves=30, random_state=4936, reg_alpha=10, reg_lambda=2)""")
    
    
    st.write("""<p style='text-align: justify; font-size: 16px'>The chart below also shows that we have optimised the model to maximise the F1 score \
        and the changes in the recall and precision scores as we have tuned the model.
             """, unsafe_allow_html=True)    
    
    img12 = Image.open("images/Discrimination_1_week.png")
    st.image(img12, width = 700) 
    
    st.write("""<p style='text-align: justify; font-size: 16px'>Once we were comfortable with the tuning, we moved on the analyse various elements of the model, starting with observing the F1 score \
        across the train and test datasets to identify any overfitting.
             """, unsafe_allow_html=True) 
    
    st.code("""
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Test F1 Score: " + str(f1_score(y_test, y_pred, average='macro')))
print("Train F1 Score: " + str(f1_score(y_train, clf.predict(X_train), average='macro')))

print("Test Recall Score: " + str(recall_score(y_test, y_pred, average='macro')))
print("Train Recall Score: " + str(recall_score(y_train, clf.predict(X_train), average='macro')))

print("Test Precision Score: " + str(precision_score(y_test, y_pred, average='macro')))
print("Train Precision Score: " + str(precision_score(y_train, clf.predict(X_train), average='macro')))
""")
    
    df = pd.DataFrame(columns=['Train_Test', 'Metric', 'Value'])  
    
    df['Train_Test'] = ['Test', 'Train', 'Test', 'Train', 'Test', 'Train']
    df['Metric'] = ['F1 Score', 'F1 Score', 'Recall', 'Recall', 'Precision', 'Precision']
    df['Value'] = [0.4530585770981545, 0.541175154259169, 0.7080815972984309, 0.7786046926686957, 0.5375662736170564, 0.5476529766788258]
    
    df
    
    fig = px.bar(df, x='Metric', y='Value', color='Train_Test', barmode='group')
    
    st.plotly_chart(fig)
    
    st.write("""<p style='text-align: justify; font-size: 16px'>As shown in the chart and table above, there is still some overfitting to the training data. \
             This is also shown below in the Learning Curve, \
            where the cross validation scores are consistently lower than the training scores across all training instances. 
            In future iterations of this work we could address this issue via some form of regularization or employing an early stopping technique. \
            For now, we will proceed with analysing this model.
             """, unsafe_allow_html=True)
    
    img12 = Image.open("images/Learning Curves for 1 Week Model.png")
    st.image(img12, width = 700)
    
    st.write("""<p style='text-align: justify; font-size: 16px'>Firstly, below are top 10 most important features in the model.
             """, unsafe_allow_html=True)
    
    img12 = Image.open("images/1 Week Feature Importance.png")
    st.image(img12, width = 700)
    
    st.write("""<p style='text-align: justify; font-size: 16px'>Weeks since last injury is the most important feature in this model (and was the same for \
             most of the longer horizon models). This seems to make intuitive sense, as if you have gove injured recently, there is probably a higher chance \
                 of injury immediately after sustaining and recovering from another injury. This is also demostrated in the chart below with ~50% of injuries \
                     occuring within 20 weeks of the previous injury and ~80% of injuries happening within a year of the previous injury.
             """, unsafe_allow_html=True)  
    
    df = pd.read_parquet('dataframes_blog/example_weeks_since_injury.parquet')
    df['Cumulative Injuries in 1 Week'] = df['injured_in_1_week'].cumsum()
    df['Cumulative Injuries in 1 Week (%)'] = 100*df['Cumulative Injuries in 1 Week']/df['injured_in_1_week'].sum()
    
    fig = px.scatter(df, x='weeks_since_last_injury', y='Cumulative Injuries in 1 Week (%)')
    fig.add_shape(go.layout.Shape(type="rect", x0 = 0, y0=0, x1=52, y1=80, opacity=0.2, fillcolor='blue', 
                                  line=dict(color='blue', dash='dash')))
    # fig.update_traces(textposition='top center')
    # fig.update_layout(title_text='80%')
    fig.add_annotation(dict(x=26, y=85, text='80% of Injuries', showarrow=True, arrowsize=3.0, ax=0, ay=-20))
    st.plotly_chart(fig)
    
    st.write("""<p style='text-align: justify; font-size: 16px'>Moving on to how well the model classifies injuries, below is the confusion matrix: 
             """, unsafe_allow_html=True)  
    
    img13 = Image.open("images/Confusion Matrix One Week.png")
    st.image(img13, width = 500)
    
    st.write("""<p style='text-align: justify; font-size: 16px'>You can see that that model misclassifies a lot of non-injuries (91543 false positives) and \
        a smaller proportion of non-injuries are classified as injuries (2007 false negatives).         
             """, unsafe_allow_html=True)
    
    images = ['images/Classification Report.png', 'images/Error Report_1_Week.png']
    st.image(images, use_column_width=True)
    
    st.write("""<p style='text-align: justify; font-size: 16px'>This point is further emphasized in the classification report and also the error report below. \
        There are misclassifictaions both on the injury and non-injuries side. Let's use the predict function on both the training and test dataset and plot the \
            predicted cumulative injuries over time when compared with actual injuries.
             """, unsafe_allow_html=True)
    
    df = pd.read_parquet('dataframes_blog/actual_vs_predicted_1_week.parquet')
    df.rename(columns={'cum_week':'Week of Career'}, inplace=True)
    
    fig = px.line(df[df['Week of Career'] > 0], x="Week of Career", y="Cumulative Injuries", title='Actual vs Predicted Injuries', color='Actual_Predicted_Train_Test')
    st.plotly_chart(fig)
    
    st.write("""<p style='text-align: justify; font-size: 16px'>As expected, the model consistently underpredicts when compared with the actual injuries for the same time period. \
        The test set starts later than the training set as we have taken the last 25% of a player's career in the test set. However, since players are generally older in the test set, \
            the number of injuries are proportionately higher in the test set when compared to the training set. What is encouraging from this analysis is that despite the model underpredicting, \
                the trend of the cumulative predicted injuries closely follows that of the actual injuries suggesting that the model is picking enough of a signal \
                    from the data to be able to predict overall trends.
            """, unsafe_allow_html=True)
    
    st.subheader('Injury Duration Prediction')
    
    st.write("""<p style='text-align: justify; font-size: 16px'>As mentioned earlier, we did also construct an injury duration prediction model which \
        looked at how long a player is expected to be injured when they incur a certain injury. In order to do this, we first created dummy variable columns \
            for the top 30 injuries across the whole dataset and added them to the existing features already in the dataset.
            """, unsafe_allow_html=True)
    
    st.code("""
feature_list = list(dict(dataset_injury['Injury'].value_counts()).keys())

for col in feature_list:
    dataset_injury.loc[dataset_injury['Injury'] == col, col] = 1
    dataset_injury.loc[dataset_injury['Injury'] != col, col] = 0)

extended_features = ['Height', 'Weight', 'cum_week', 'defender', 'attacker', 'midfielder', 'goalkeeper', 'right_foot', 'cum_injury_total',
 'weeks_since_last_injury', 'Min_cum', 'Gls_cum', 'Ast_cum', 'PK_cum', 'PKatt_cum', 'Sh_cum', 'SoT_cum', 'CrdY_cum',
 'CrdR_cum', 'Touches_cum', 'Press_cum', 'Tkl_cum', 'Int_cum', 'Blocks_cum', 'xG_cum', 'npxG_cum', 'xA_cum',
 'SCA_cum', 'GCA_cum', 'Cmp_cum', 'Att_cum', 'Prog_cum', 'Carries_cum', 'Prog.1_cum', 'Succ_cum', 'Att.1_cum',
 'Fls_cum', 'Fld_cum', 'Off_cum', 'Crs_cum', 'TklW_cum', 'OG_cum', 'PKwon_cum', 'PKcon_cum', 'Serie A_cum', 'Premier League_cum',
 'La Liga_cum', 'Ligue 1_cum', 'Bundesliga_cum', 'Champions Lg_cum', 'Europa Lg_cum', 'FIFA World Cup_cum',
 'UEFA Nations League_cum', 'UEFA Euro_cum', 'Copa América_cum', 'Away_cum', 'Home_cum', 'Neutral_cum',
                     'Injury_Recurrence_cum'] + list(dict(dataset_injury['Injury'].value_counts()).keys())[:30]
"""
    )
    
    st.write("""<p style='text-align: justify; font-size: 16px'>We also created a recurrent injury column, which looks at how often each player gets a \
        certain type of injury over their career.""", unsafe_allow_html=True)
    
    st.code("""
injury_recurrence = pd.DataFrame(dataset_injury.groupby(['FBRefID','Injury', 'injury_week', 'injury_year'])['Injury'].count()).rename(columns={'Injury':'Injury_Recurrence'}).reset_index().sort_values(by=['FBRefID','injury_year', 'injury_week'])
injury_recurrence['Injury_Recurrence_cum'] = injury_recurrence.groupby(['FBRefID','Injury'])['Injury_Recurrence'].cumsum()
""")
    
    st.write("""<p style='text-align: justify; font-size: 16px'>Then we followed a similar workflow as the classification modelling with PyCaret but with the \
        Regression module.""", unsafe_allow_html=True)
    
    st.code("""
exp_reg = setup(dataset_injury_1[extended_features + [injured_pred]], target = injured_pred, fold=10, feature_selection=True)
best_model = compare_models(turbo=False)
""")
    
    images = 'images/Injury Duration Models.png'
    st.image(images, use_column_width=True)

    st.write("""<p style='text-align: justify; font-size: 16px'> The best R^2 score was 18.84% using the <b>Bayesian Ridge</b> model.\
        We did not pursue any further analysis on this model in this iteration for two reasons. Firstly, the duration which a player gets an injury is \
            highly unique to the player's physiology and their rehabilitation program. This is information which we don't currently have and this is relfected \
                in the accuracy scores. Second, injury duration is something that can currently be advised by team doctors, and hence isn't as valuable as \
                    predicting injuries before they happen. We may further tune and improve this model in the future once further data is acquired.""", unsafe_allow_html=True)

    st.subheader('Final Thoughts and Next Steps')
    
    st.write("""<p style='text-align: justify; font-size: 16px'>As demonstrated by the model accuracies, predicting whether injuries occur is a really difficult task. \
        There are some issues with overfitting which can be reduced as explained earlier. \
        The data acquired for this analysis was mainly game statistics and some player profile information. However, a lot of injuries are also acquired due to \
            players activities outside of competitive games (i.e. in training). This information is generally not publicly available. 
            
            \nAnother piece of \
                information that we have not taken into account is biometric data of players. As mentioned in the book, <i>"Why we Sleep"<i> by Matthew Walker, \
                    the occurence of injuries has been linked to lack of sleep. Other information such as the amount of running a player does on the field \
                        could also be captured via GPS tracking devices. Again, this information is generally not publicly available but club would maintain \
                            databases on their players related to this. If this information can be obtained, the accuracy of the models will be expected to \
                                further improve.""", unsafe_allow_html=True)
    
    st.write("""<p style='text-align: justify; font-size: 16px'>In the next blog post, we will introduce our Injury Prediction plots, and later our Interactive Injury Prediction Tool \
        where you can play around with the predictions to determine whether your favourite player is in danger of getting an injury in the near future.""", unsafe_allow_html=True)

###############################################################################################################################

# SECTION: INJURY PREDICTION TOOL
elif section == "Injury Prediction":

    imgballinfield = Image.open("images/ballinfield.jpg")
    st.image(imgballinfield, width = 700)

    st.header('Injury Prediction')

    st.write("<p style='text-align: justify; font-size: 16px'>As we have mentioned before, there have been 5 previous stages before being able to do any kind of prediction. \
        Scraping of data from the web, data manipulation, feature engineering, visual exploration of data and model building, all gave \
        us the best models to predict injuries.</h1>", unsafe_allow_html=True)
    
    st.write("<p style='text-align: justify; font-size: 16px'>To start our prediction process we broke up the task into prediction horizons. We decided to predict if a player would \
        get injured during the next week, month, quarter, semester, or year. Our model numbers showed that as our prediction was further away, \
        it was harder to predict with accuracy if a player would get injured.  The best horizon was the week horizon with an F1-score \
        of .41 using the Light Gradient Boosting Machine.</h1>", unsafe_allow_html=True)

    st.write("<p style='text-align: justify; font-size: 16px'>Once all the models were trained, they were saved into a pickle file to retrieve later. Turns out there were different \
        models in different horizons.  For example, as we mentioned before the one-week horizon had the Light Gradient Boosting Machine \
        as its best performing model. The 1-semester horizon had the Ada Boost Classifier as its best performing model. Producing \
        predictions was a fairly simple process after we had finished all the preceding tasks.  We loaded the models and fired up a \
        prediction for all the values in our data set.  Then we accounted for the predicted injuries that had an injury the week before. \
        That is if the model predicted an injury the week before we would reset the next week to zero assuming that an already injured \
        player could not get injured again.  Once we had these numbers we accumulated them in a single column and accounted for the time \
        window we were predicting.  These predicted values were finally combined with the real values of the dataset to create a continuous \
        time series with a line with two colors, blue for real injuries and orange for predicted injuries. We decided to do \
        this to aid the viewer in detecting injuries and making inferences from them.</h1>", unsafe_allow_html=True)

    week = Image.open('images/week.png')
    st.image(week, width = 800)

    st.write("<p style='text-align: justify; font-size: 16px'> On the first visualization, we observe Neymar's injuries history, which is \
        represented by the blue line, along with our injury prediction for the following week, which is displayed in orange. Since the line \
        has not experienced an increase, our system is predicting that Neymar will not get injured during the next week.</h1>", unsafe_allow_html=True)
    # st.write('Our system predicts that Neymar will not get injured during the next week.')

    # month = Image.open('images/month.png')
    # st.image(month, width = 800)
    
    quarter = Image.open('images/quarter.png')
    st.image(quarter, width = 800)

    st.write("<p style='text-align: justify; font-size: 16px'>This visualization is making a quarter prediction(12 weeks). We can already see \
        that the orange line is not straight anymore when comparing it to the previous graph. The quarter window injury prediction is informing \
        us that Neymar will suffer one injury in the next 12 weeks. This injury will happen in around 7 weeks.</h1>", unsafe_allow_html=True)

    semester = Image.open('images/semester.png')
    st.image(semester, width=800)

    st.write("<p style='text-align: justify; font-size: 16px'>This graph will give us a 1 semester (26 weeks) injury prediction. This is similar \
        to our visualization above, where Neymar experienced one injury after 7 weeks. Here, Neymar also experienced a single injury, but if we \
        take a closer look, our prediction tells us that Neymar will get injured in around 15 weeks.</h1>", unsafe_allow_html=True)
    
    year = Image.open('images/year.png')
    st.image(year, width=800)

    st.write("<p style='text-align: justify; font-size: 16px'>Finally, the one-year prediction yields that Neymar is going to get injured \
        3 times in the next year, where the first injury occurs only after a few weeks.</h1>", unsafe_allow_html=True)

    st.write("<p style='text-align: justify; font-size: 16px'>Please use our interactive prediction tool which is located two sections down.</h1>", unsafe_allow_html=True)

###############################################################################################################################
  
elif section == "Exploration Tool (BETA)":

    imgsoccer = Image.open("images/soccer.jpg")
    st.image(imgsoccer, width = 700)
    
    st.header('Interactive Exploration Tool (BETA)')
    st.write('Please feel free to try out our interactive exploration tool!')
    
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

###############################################################################################################################

elif section == "Injury Prediction Tool (BETA)":
    
    imgfield = Image.open("images/fields.jpg")
    st.image(imgfield, width = 700)
    
    st.header("Interactive Injury Prediction Tool (BETA)")
    st.write('Please feel free to try out our interactive prediction tool!')

    st.write('* (sample dataset used for performance purposes)')

    @st.cache(allow_output_mutation = True)
    def get_inter_pred_data():
        df = dd.read_parquet('dataframes_blog/small_dataset_with_prediction.parquet')
        return df

    df_pred_int = get_inter_pred_data().compute()
    
    st.subheader('Pick a player to predict his injuries.')

    sorted_unique_player = df_pred_int['name'].unique()
    player_to_forecast = st.selectbox('Select player to forecast (type or choose):',sorted_unique_player)

    picked_player_to_forecast_df = df_pred_int[df_pred_int['name'] == player_to_forecast]

    # Function to Forecast players injuries and plot a chart
    def forecast_it(weeks, df1):
        df = df1.reset_index()
        final_index = df.index.max()
        pos = 'cum_injury_in_'+str(weeks)+'_week'
        for n in range(weeks,0,-1): 
            df.loc[final_index-n+1, pos] = ''
        df.loc[df[pos] == "", 'single_injury_prediction_in_'+str(weeks)+'_week'] = df[pos].shift(1) 
        df.loc[df['single_injury_prediction_in_'+str(weeks)+'_week'] == "", 'single_injury_prediction_in_'+str(weeks)+'_week'] = \
            df['unique_predicted_injuries_'+str(weeks)+'_week']
        df['cum_injury_prediction_in_'+str(weeks)+'_week'] = df['single_injury_prediction_in_'+str(weeks)+'_week'].cumsum()
        # df['cum_injury_prediction_in_'+str(weeks)+'_week'] = df['cum_injury_prediction_in_'+str(weeks)+'_week'].fillna('')
        # df.loc[df['cum_injury_in_'+str(weeks)+'_week'] != "", 'cum_injury_prediction_in_'+str(weeks)+'_week'] = df['cum_injury_in_'+str(weeks)+'_week']
        df.loc[df['cum_injury_prediction_in_'+str(weeks)+'_week'].isna(), 'cum_injury_prediction_in_'+str(weeks)+'_week'] = df['cum_injury_in_'+str(weeks)+'_week']

        df.loc[df['cum_injury_in_'+str(weeks)+'_week'] != "", 'type_for_'+str(weeks)] = 'Actual Injuries'
        df.loc[df['cum_injury_in_'+str(weeks)+'_week'] == "", 'type_for_'+str(weeks)] = 'Predicted Injuries'

        extra_point = df[df['type_for_'+str(weeks)] == 'Actual Injuries'].tail(1)
        extra_point['type_for_'+str(weeks)] = 'Predicted Injuries'
        df = pd.concat([df, extra_point]) #.reset_index(drop=True)

        df.loc[df['attacker'] == 1, 'position'] = 'attacker'
        df.loc[df['defender'] == 1, 'position'] = 'defender'
        df.loc[df['goalkeeper'] == 1, 'position'] = 'goalkeeper'
        df.loc[df['midfielder'] == 1, 'position'] = 'midfielder'

        df = df[['FBRefID', 'name', 'date', 'cum_week', 'position', 'Min', 'cum_injury_prediction_in_'+str(weeks)+'_week', 'type_for_'+str(weeks)]]
        
        df.loc[df['cum_injury_prediction_in_'+str(weeks)+'_week'] == '', 'cum_injury_prediction_in_'+str(weeks)+'_week'] = None
        df['cum_injury_prediction_in_'+str(weeks)+'_week'] = df['cum_injury_prediction_in_'+str(weeks)+'_week'].ffill()
        
        return df
    
    # Forecasting the different horizons for picked player
    final_small_df_1_week = forecast_it(1, picked_player_to_forecast_df)
    final_small_df_4_week = forecast_it(4, picked_player_to_forecast_df)
    final_small_df_12_week = forecast_it(12, picked_player_to_forecast_df)
    final_small_df_26_week = forecast_it(26, picked_player_to_forecast_df)
    final_small_df_52_week = forecast_it(52, picked_player_to_forecast_df)

    # Function to change name of columns of all dataframes
    def transform_it(weeks, df):
        df = df.rename(columns={'cum_injury_prediction_in_'+str(weeks)+'_week':'predicted_cum_injuries', 'type_for_'+str(weeks):'label'})
        df['horizon'] = weeks
        return df

    # Changing column names and concatenating dataframes
    final_df_lst = [final_small_df_1_week, final_small_df_4_week, final_small_df_12_week, final_small_df_26_week, \
        final_small_df_52_week]
    weeks_lst = [1, 4, 12, 26, 52]

    final_player_df = pd.DataFrame(columns=list(transform_it(1,final_small_df_1_week).columns))
    counter = 0

    for df in final_df_lst:
        final_player_df = final_player_df.append(transform_it(weeks_lst[counter], df))
        counter += 1

    #Plotting Chart
    @st.cache(allow_output_mutation = True)
    def plot_it(weeks, df):
        
        df = df[df['horizon'] == weeks]

        if weeks == 1:
            df_final = df[len(df)-25:].sort_index()
        elif weeks == 4:
            df_final = df[len(df)-25:].sort_index()
        elif weeks == 12:
            df_final = df[len(df)-25:].sort_index()
        else:
            df_final = df[len(df)-(weeks*2):].sort_index()

        domain_min = df_final['predicted_cum_injuries'].min()-5
        domain_max = df_final['predicted_cum_injuries'].max()+5

        chart = alt.Chart(df_final).mark_line().encode(x=alt.X('date', timeUnit='yearmonthdate'), \
            y=alt.Y('predicted_cum_injuries', scale=alt.Scale(domain=[domain_min, domain_max])), color=alt.Color('label'), \
            strokeDash=alt.condition(alt.datum.symbol == 'label', alt.value([5, 5]), alt.value([0]))). \
            properties(title = str(df_final['name'].iloc[0])+"'s Injury Prediction for "+str(weeks)+" weeks", width=800, height=300)

        return chart
    
    horizons = ['1 week', '1 month', '1 quarter', '1 semester', '1 year']
    player_horizon = st.selectbox('Please select player horizon:',horizons)

    avg_playing_time = final_player_df['Min'].mean()

    if player_horizon == '1 week':
        horizon = 1
    elif player_horizon == '1 month':
        horizon = 4
    elif player_horizon == '1 quarter':
        horizon = 12
    elif player_horizon == '1 semester':
        horizon = 26
    else:
        horizon = 52

    df_injuries = final_player_df[final_player_df['horizon'] == horizon]
    min_value = df_injuries[df_injuries['label'] == 'Actual Injuries'][-2:]['predicted_cum_injuries'].iloc[0]
    max_value = df_injuries[df_injuries['label'] == 'Predicted Injuries'][-2:]['predicted_cum_injuries'].iloc[0]

    num_injuries = int(max_value - min_value)

    # Doing some outlier cleaning
    if player_to_forecast == 'Lionel Messi':
        final_player_df = final_player_df.drop(labels=758, axis=0)
    if player_to_forecast == 'Gareth Bale':
        final_player_df = final_player_df.drop(labels=655, axis=0)

    # Adjusting text for players who have just gotten injured the month before
    if (player_to_forecast == 'Gareth Bale') & (horizon == 1):
        num_injuries = num_injuries - 1
    if (player_to_forecast == 'Gareth Bale') & (horizon == 4):
        num_injuries = num_injuries - 1
    if (player_to_forecast == 'Gareth Bale') & (horizon == 12):
        num_injuries = num_injuries - 1
    if (player_to_forecast == 'Gareth Bale') & (horizon == 26):
        num_injuries = num_injuries - 1
    if player_to_forecast == 'Virgil van Dijk':
        num_injuries = num_injuries - 1
    if (player_to_forecast == 'Sergio Ramos') & (horizon == 4):
        num_injuries = num_injuries - 1
    if (player_to_forecast == 'Sergio Ramos') & (horizon == 12):
        num_injuries = num_injuries - 1
    if (player_to_forecast == 'Sergio Ramos') & (horizon == 26):
        num_injuries = num_injuries - 1
    if (player_to_forecast == 'Sergio Ramos') & (horizon == 52):
        num_injuries = num_injuries - 1
    if (player_to_forecast == 'Paul Pogba') & (horizon == 1):
        num_injuries = num_injuries - 1
    if (player_to_forecast == 'Paul Pogba') & (horizon == 4):
        num_injuries = num_injuries - 1
    if (player_to_forecast == 'Paul Pogba') & (horizon == 12):
        num_injuries = num_injuries - 1
    if (player_to_forecast == 'Paul Pogba') & (horizon == 26):
        num_injuries = num_injuries - 1
    if (player_to_forecast == 'Joshua Kimmich') & (horizon == 1):
        num_injuries = num_injuries - 1
    if (player_to_forecast == 'Joshua Kimmich') & (horizon == 4):
        num_injuries = num_injuries - 1
    if (player_to_forecast == 'Joshua Kimmich') & (horizon == 12):
        num_injuries = num_injuries - 1
    if (player_to_forecast == 'Joshua Kimmich') & (horizon == 26):
        num_injuries = num_injuries - 1
    

    
    st.write(str(final_player_df['name'].iloc[0])+"'s position is "+final_player_df['position'].iloc[0]+"! He plays and average \
    of "+str(round(avg_playing_time))+" minutes of competitive soccer per week. You picked a horizon of "+str(horizon)+ \
    ' weeks. According to our model, during the next '+str(horizon)+' weeks, '+str(final_player_df['name'].iloc[0]+' is going to \
    get '+str(num_injuries)+' injuries.'))

    chart_output = copy.deepcopy(plot_it(horizon, final_player_df))    
    st.altair_chart(chart_output, use_container_width=False)


###############################################################################################################################

elif section == "Conclusion" :
    st.header("Conclusion, Challenges, and Future Work")
 
    st.subheader("Conclusion")
    st.write("<p style='text-align: justify; font-size: 16px'>In today’s football, players compete more and moreover a single year. Apart from playing more than one \
        tournament for their clubs, who own the rights for the players, the most talented players must attend \
        international duties with their national teams, as well as friendly matches with both their clubs and \
        countries. It is normal to see footballers get injured from time to time since this sport requires physical \
        demand and players are constantly being tackled and exposed to full body or kick collisions. Additionally, \
        players have to go through rigorous training and do not have much time to rest. Our goal was to create \
        a machine learning model that could predict when a player will get injured......</h1>", unsafe_allow_html=True)

    st.write("<p style='text-align: justify; font-size: 16px'>The project has been a great experience for all of us. Not only because we were challenged to come up \
        with a data science project that could be applied to the real world and also learned a great number of \
        new tools we previously did not have any knowledge about, but most importantly because this capstone \
        gave all of us the opportunity to work with one another as data scientists. We all faced many challenges, \
        encountered multiple roadblocks, stayed up long hours, but we also discovered new capabilities together \
        and supported each other at all times. We all agree that this has been the greatest takeaway from the \
        course and even the entire program. It was definitely not a simple task to complete .....</h1>", unsafe_allow_html=True)
    
    st.subheader("Challenges")
    st.write("")
    img25 = Image.open("images/image25.png", )
    st.image(img25, use_column_width ="always")
    
    st.write("**Data Scrapping**")
    st.write("<p style='text-align: justify; font-size: 16px'>There was a considerable number of roadblocks in data scrapping, and we did not envision the challenges before we \
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
    st.write("<p style='text-align: justify; font-size: 16px'>First, it took some effort to locally install and integrate VS Code and GitHub together. After the installation finished,\
        and the repository was set and authenticated, we were capable to begin working in parallel, which we all enjoyed. However, we did \
        undergo some minor problems when submitting new changes and pulling new requests. This mainly occurred because we failed to \
        communicate which notebooks were being updated, as well as not pulling on time when needed to. This resulted in many conflicts \
        that were not straightforward to solve, which even cause resetting the repositories in some cases. Once we were all accustomed to it, \
        we did not face major problems.</h1>", unsafe_allow_html=True)
    st.write("")
    st.write("**Amazon Web Services**")
    st.write("<p style='text-align: justify; font-size: 16px'>In addition to using VS Code to work with our GitHub repository, we incorporated Amazon Web Services so that we can attempt\
        to run scripts that required a great amount of memory usage. The process of setting AWS was lengthy, and although we found guidelines\
        on how to install and integrate AWS into our machines, it was still confusing to deal with it. Once AWS was installed and running, \
        all of us started to experience problems with the connection during the web scrapping phase. This occurred regularly, thus we \
        decided to avoid employing AWS for the data scrapping..</h1>", unsafe_allow_html=True)
    st.write("")
    st.write("**Time Series Model**")
    st.write("<p style='text-align: justify; font-size: 16px'>Building our time series dataset was not an easy task. It required a great amount of time to examine and debate how the model \
        was going to be built as well as what features had to be included, and what data instances we had to drop. It took multiple attempts \
        to construct the desired data frame, and it also required us to investigate the actual machine learning algorithms we were going to employ \
        before we prepare the final dataset..</h1>", unsafe_allow_html=True)
    st.write("")
    st.write("**StreamLit**")
    st.write("<p style='text-align: justify; font-size: 16px'>This python library allowed us to build our blog for the audience. It did not take a great amount of research since \
        it was fairly easy to understand compared to other libraries we have used before, and also Streamlit integrated amazingly \
        well with other libraries such Pandas or Altair. However, the challenge was when building our Players’ comparison tool. \
        Even though we achieved to develop our tools and displayed them through Streamlit, it was taken a very long time for the \
        custom apps to run because of the size of our main dataset and the magnitude of our models. We wanted to avoid creating \
        frustration between our users, so we decided to take a different approach. First, we converted our main data frame from \
        a CSVfile to a parquet file. This significantly reduced the weight of the dataset; nevertheless, the tool barely improve.</h1>", unsafe_allow_html=True)

    st.subheader("Future Work")
    
    st.write("<p style='text-align: justify; font-size: 16px'>We have collected a significant amount of data that contains players’ \
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
    
    
    ###############################################################################################################################
    
else:
    
    # st.header("Statement of Work and Citations")
 
    st.subheader("Statement of Work")

    st.write("<p style='text-align: justify; font-size: 16px'>The members of this project were separated by 15 hours due to their locations \
        (Australia, North America, South America). To complete this project each team member was assigned an iteration of every step. This way \
        all team members participated in every step of the way and contributed to the whole process being aware of the steps taken in each iteration.\
        All team members participated and presented in every single stand-up as well as all weekly status updates. The project blog/website was \
        designed and proofread by all team participants. Collaboration was achieved with Visual Studio Code via Jupyter notebooks. \
        In addition, the presentation and project overview video were previously discussed and produced among team members. For communication \
        purposes, text messaging and conference calls were employed via personal cellphone and Zoom video calls with screen sharing.</h1>", unsafe_allow_html=True)

    st.write("<p style='text-align: justify; font-size: 16px'>The project has been a fantastic experience for all of us. Not only because we\
        were challenged to come up with a data science project that could be applied to the real world and also learned a great number of new\
        tools, we previously did not have any knowledge about. Most importantly, this capstone allowed all of us to work with one another as\
        data scientists. We all faced many challenges, encountered multiple roadblocks, stayed up long hours, but we also discovered new\
        capabilities together and supported each other at all times. We all agree that this has been the greatest takeaway from the\
        course and the entire MADS program. It was not a simple task to complete, but we can now say that the reward is greater\
        than the sacrifice.</h1>", unsafe_allow_html=True)
    
    st.write("*Sravan Rajupalem, Renzo Maldonado, Victor Ruiz*")
    
    st.subheader("Citations")
     
    st.write("FBref.com. 2021. Football Statistics and History | FBref.com. [online] Available at: <https://fbref.com/en/>")
    st.write("Transfermarkt.com. 2021. Football transfers, rumours, market values and news. [online] Available at: <https://www.transfermarkt.com/>")
    st.write("Medium. 2021. Beating soccer odds using Machine Learning — Project Walkthrough. [online] Available at: <https://medium.com/analytics-vidhya/beating-soccer-odds-using-machine-learning-project-walkthrough-a1c3445b285a>")
    st.write("Kaggle.com. 2021. Comprehensive data exploration with Python. [online] Available at: <https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python>")
    st.write("Medium. 2021. Creating Multipage applications using Streamlit (efficiently!). [online] Available at: <https://towardsdatascience.com/creating-multipage-applications-using-streamlit-efficiently-b58a58134030>")