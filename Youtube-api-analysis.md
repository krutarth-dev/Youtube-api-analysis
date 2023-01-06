```python
api_key="AIzaSyBW8NEH3lkriNiX4Xh4ZcqKDSiHRGG8i7s"
```


```python
from googleapiclient.discovery import build
import pandas as pd
from dateutil import parser
from IPython.display import JSON

# Data viz packages
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from wordcloud import WordCloud
```

    [nltk_data] Downloading package stopwords to /Users/apple/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to /Users/apple/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!



```python
channel_ids=['UC7_YxT-KID8kRbqZo7MyscQ',
            #more channels
            ]
```


```python
api_service_name = "youtube"
api_version = "v3"

# Get credentials and create an API client
youtube = build(
    api_service_name, api_version, developerKey=api_key)

```


```python
def get_channel_stats(youtube, channel_ids):
    
    all_data = []
    
    request = youtube.channels().list(
        part="snippet,contentDetails,statistics",
        id=','.join(channel_ids)
    )
    response = request.execute()

    # loop through items
    for item in response['items']:
        data = {'channelName': item['snippet']['title'],
                'subscribers': item['statistics']['subscriberCount'],
                'views': item['statistics']['viewCount'],
                'totalVideos': item['statistics']['videoCount'],
                'playlistId': item['contentDetails']['relatedPlaylists']['uploads']
               }
        all_data.append(data)
    return(pd.DataFrame(all_data))
```


```python
channel_stats = get_channel_stats(youtube, channel_ids)
```


```python
channel_stats
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>channelName</th>
      <th>subscribers</th>
      <th>views</th>
      <th>totalVideos</th>
      <th>playlistId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Markiplier</td>
      <td>34200000</td>
      <td>19425422867</td>
      <td>5381</td>
      <td>UU7_YxT-KID8kRbqZo7MyscQ</td>
    </tr>
  </tbody>
</table>
</div>




```python
playlist_id = ["UU7_YxT-KID8kRbqZo7MyscQ","UUsXVk37bltHxD1rDPwtNM8Q"]

def get_video_ids(youtube,playlist_id): 
    
    request = youtube.playlistItems().list(
        part="snippet,contentDetails",
        playlistId=playlist_id,
        maxResults = 50
    )
    response = request.execute()
    
    for item in response['items']:
        video_ids.append(item['contentDetails']['videoId'])
        
    next_page_token = response.get('nextPageToken')
    while next_page_token is not None:
        request = youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=playlist_id,
            maxResults = 50
        )
        response = request.execute()
    
        for item in response['items']:
            video_ids.append(item['contentDetails']['videoId'])
            
        next_page_token = response.get('nextPageToken')
            
    return video_ids
```


```python
#get vedio IDs
video_ids = get_video_ids(youtube, playlist_id)
```


    ---------------------------------------------------------------------------

    HttpError                                 Traceback (most recent call last)

    <ipython-input-422-b4bf94c6d9f7> in <module>
          1 #get vedio IDs
    ----> 2 video_ids = get_video_ids(youtube, playlist_id)
    

    <ipython-input-421-2a0dc768a9a3> in get_video_ids(youtube, playlist_id)
          8         maxResults = 50
          9     )
    ---> 10     response = request.execute()
         11 
         12     for item in response['items']:


    ~/opt/anaconda3/lib/python3.8/site-packages/googleapiclient/_helpers.py in positional_wrapper(*args, **kwargs)
        128                 elif positional_parameters_enforcement == POSITIONAL_WARNING:
        129                     logger.warning(message)
    --> 130             return wrapped(*args, **kwargs)
        131 
        132         return positional_wrapper


    ~/opt/anaconda3/lib/python3.8/site-packages/googleapiclient/http.py in execute(self, http, num_retries)
        936             callback(resp)
        937         if resp.status >= 300:
    --> 938             raise HttpError(resp, content, uri=self.uri)
        939         return self.postproc(resp, content)
        940 


    HttpError: <HttpError 404 when requesting https://youtube.googleapis.com/youtube/v3/playlistItems?part=snippet%2CcontentDetails&playlistId=%5B%27UU7_YxT-KID8kRbqZo7MyscQ%27%2C+%27UUsXVk37bltHxD1rDPwtNM8Q%27%5D&maxResults=50&key=AIzaSyBW8NEH3lkriNiX4Xh4ZcqKDSiHRGG8i7s&alt=json returned "The playlist identified with the request's <code>playlistId</code> parameter cannot be found.". Details: "[{'message': "The playlist identified with the request's <code>playlistId</code> parameter cannot be found.", 'domain': 'youtube.playlistItem', 'reason': 'playlistNotFound', 'location': 'playlistId', 'locationType': 'parameter'}]">



```python
len(video_ids)
```




    53455




```python
def get_video_details(youtube, video_ids):
    
    all_video_info = []
    
    for i in range(0, len(video_ids),50):
        request = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        id=video_ids[0:50]
    )
    response = request.execute()
    
    for video in response ['items']:
            stats_to_keep = {'snippet':['channelTitle','title','description','tags','publishedAt'],
                         'statistics':['viewCount','likeCount','favouriteCount','commentCount'],
                         'contentDetails':['duration','definition','caption']
                        }
            video_info = {}
            video_info['video_id'] = video['id']
        
            for k in stats_to_keep.keys():
                for v in stats_to_keep[k]:
                    try:
                        video_info[v] = video[k][v]
                    except:
                        video_info[v] = None
            
            all_video_info.append(video_info)
        
    return pd.DataFrame(all_video_info)
```


```python
# get vedio details
ideo_df = get_video_details(youtube, video_ids)
video_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>video_id</th>
      <th>channelTitle</th>
      <th>title</th>
      <th>description</th>
      <th>tags</th>
      <th>publishedAt</th>
      <th>viewCount</th>
      <th>likeCount</th>
      <th>favouriteCount</th>
      <th>commentCount</th>
      <th>duration</th>
      <th>definition</th>
      <th>caption</th>
      <th>publishedDayName</th>
      <th>durationSecs</th>
      <th>tagCount</th>
      <th>title_no_stopwords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>VN8pnrs1jBE</td>
      <td>Markiplier</td>
      <td>IT'S ENDING ME!! | Endoparasitic - Part 3</td>
      <td>This is the ending of Endoparasitic! I hope we...</td>
      <td>None</td>
      <td>2023-01-03 17:00:14+00:00</td>
      <td>1201399.0</td>
      <td>70034.0</td>
      <td>NaN</td>
      <td>3786.0</td>
      <td>PT29M7S</td>
      <td>hd</td>
      <td>false</td>
      <td>Tuesday</td>
      <td>1747.0</td>
      <td>0</td>
      <td>[IT'S, ENDING, ME!!, |, Endoparasitic, -, Part...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0ALfWW_U728</td>
      <td>Markiplier</td>
      <td>My Mom Plays Five Nights at Freddy's: Sister L...</td>
      <td>Markiplier's Mom attempts to beat Five Nights ...</td>
      <td>[markiplier, fnaf, five nights at freddy's, fi...</td>
      <td>2023-01-02 17:35:18+00:00</td>
      <td>1998009.0</td>
      <td>192801.0</td>
      <td>NaN</td>
      <td>12319.0</td>
      <td>PT24M56S</td>
      <td>hd</td>
      <td>false</td>
      <td>Monday</td>
      <td>1496.0</td>
      <td>22</td>
      <td>[My, Mom, Plays, Five, Nights, Freddy's:, Sist...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>W2NafoExi1Y</td>
      <td>Markiplier</td>
      <td>best raft ever… | Raft</td>
      <td>This is the best raft we've ever built. ever. ...</td>
      <td>[markiplier, raft, muyskerm, lordminion777, ma...</td>
      <td>2022-12-30 17:00:30+00:00</td>
      <td>1810070.0</td>
      <td>82449.0</td>
      <td>NaN</td>
      <td>4119.0</td>
      <td>PT36M29S</td>
      <td>hd</td>
      <td>false</td>
      <td>Friday</td>
      <td>2189.0</td>
      <td>8</td>
      <td>[best, raft, ever…, |, Raft]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5DwIJPSPtFg</td>
      <td>Markiplier</td>
      <td>The Fall of Markiplier</td>
      <td>Mark takes a tumble. Here's the story of the p...</td>
      <td>[markiplier, animated, distractible, mark bob ...</td>
      <td>2022-12-28 21:25:36+00:00</td>
      <td>1728446.0</td>
      <td>162281.0</td>
      <td>NaN</td>
      <td>5483.0</td>
      <td>PT2M7S</td>
      <td>hd</td>
      <td>false</td>
      <td>Wednesday</td>
      <td>127.0</td>
      <td>20</td>
      <td>[The, Fall, Markiplier]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>299Zd7nmqdg</td>
      <td>Markiplier</td>
      <td>...OOPS! | Shadows Over Loathing - Part 4</td>
      <td>Ah... so that's... that's not gonna come back,...</td>
      <td>[shadows over loathing, west of loathing, funn...</td>
      <td>2022-12-27 17:00:01+00:00</td>
      <td>1154311.0</td>
      <td>58126.0</td>
      <td>NaN</td>
      <td>3016.0</td>
      <td>PT50M35S</td>
      <td>hd</td>
      <td>false</td>
      <td>Tuesday</td>
      <td>3035.0</td>
      <td>12</td>
      <td>[...OOPS!, |, Shadows, Over, Loathing, -, Part...</td>
    </tr>
  </tbody>
</table>
</div>



## Data pre-processing


```python
video_df.isnull().any()
```




    video_id              False
    channelTitle          False
    title                 False
    description           False
    tags                   True
    publishedAt           False
    viewCount             False
    likeCount             False
    favouriteCount         True
    commentCount          False
    duration              False
    definition            False
    caption               False
    publishedDayName      False
    durationSecs          False
    tagCount              False
    title_no_stopwords    False
    dtype: bool




```python
video_df.dtypes
```




    video_id                               object
    channelTitle                           object
    title                                  object
    description                            object
    tags                                   object
    publishedAt           datetime64[ns, tzutc()]
    viewCount                             float64
    likeCount                             float64
    favouriteCount                        float64
    commentCount                          float64
    duration                               object
    definition                             object
    caption                                object
    publishedDayName                       object
    durationSecs                          float64
    tagCount                                int64
    title_no_stopwords                     object
    dtype: object




```python
numeric_cols = ['viewCount','likeCount','favouriteCount','commentCount']
video_df[numeric_cols] = video_df[numeric_cols].apply(pd.to_numeric,errors='coerce',axis=1)
```


```python
#publish day in the week
video_df['publishedAt']= video_df['publishedAt'].apply(lambda x: parser.parse(x))
video_df['publishedDayName']= video_df['publishedAt'].apply(lambda x: x.strftime("%A"))
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~/opt/anaconda3/lib/python3.8/site-packages/pandas/core/indexes/extension.py in map(self, mapper, na_action)
        297         try:
    --> 298             result = mapper(self)
        299 


    <ipython-input-431-50b76488ab6f> in <lambda>(x)
          1 #publish day in the week
    ----> 2 video_df['publishedAt']= video_df['publishedAt'].apply(lambda x: parser.parse(x))
          3 video_df['publishedDayName']= video_df['publishedAt'].apply(lambda x: x.strftime("%A"))


    ~/opt/anaconda3/lib/python3.8/site-packages/dateutil/parser/_parser.py in parse(timestr, parserinfo, **kwargs)
       1373     else:
    -> 1374         return DEFAULTPARSER.parse(timestr, **kwargs)
       1375 


    ~/opt/anaconda3/lib/python3.8/site-packages/dateutil/parser/_parser.py in parse(self, timestr, default, ignoretz, tzinfos, **kwargs)
        645 
    --> 646         res, skipped_tokens = self._parse(timestr, **kwargs)
        647 


    ~/opt/anaconda3/lib/python3.8/site-packages/dateutil/parser/_parser.py in _parse(self, timestr, dayfirst, yearfirst, fuzzy, fuzzy_with_tokens)
        724         res = self._result()
    --> 725         l = _timelex.split(timestr)         # Splits the timestr into tokens
        726 


    ~/opt/anaconda3/lib/python3.8/site-packages/dateutil/parser/_parser.py in split(cls, s)
        206     def split(cls, s):
    --> 207         return list(cls(s))
        208 


    ~/opt/anaconda3/lib/python3.8/site-packages/dateutil/parser/_parser.py in __init__(self, instream)
         74         elif getattr(instream, 'read', None) is None:
    ---> 75             raise TypeError('Parser must be a string or character stream, not '
         76                             '{itype}'.format(itype=instream.__class__.__name__))


    TypeError: Parser must be a string or character stream, not DatetimeIndex

    
    During handling of the above exception, another exception occurred:


    TypeError                                 Traceback (most recent call last)

    <ipython-input-431-50b76488ab6f> in <module>
          1 #publish day in the week
    ----> 2 video_df['publishedAt']= video_df['publishedAt'].apply(lambda x: parser.parse(x))
          3 video_df['publishedDayName']= video_df['publishedAt'].apply(lambda x: x.strftime("%A"))


    ~/opt/anaconda3/lib/python3.8/site-packages/pandas/core/series.py in apply(self, func, convert_dtype, args, **kwds)
       4133             if is_extension_array_dtype(self.dtype) and hasattr(self._values, "map"):
       4134                 # GH#23179 some EAs do not have `map`
    -> 4135                 mapped = self._values.map(f)
       4136             else:
       4137                 values = self.astype(object)._values


    ~/opt/anaconda3/lib/python3.8/site-packages/pandas/core/arrays/datetimelike.py in map(self, mapper)
        703         from pandas import Index
        704 
    --> 705         return Index(self).map(mapper).array
        706 
        707     def isin(self, values) -> np.ndarray:


    ~/opt/anaconda3/lib/python3.8/site-packages/pandas/core/indexes/extension.py in map(self, mapper, na_action)
        306             return result
        307         except Exception:
    --> 308             return self.astype(object).map(mapper)
        309 
        310     @doc(Index.astype)


    ~/opt/anaconda3/lib/python3.8/site-packages/pandas/core/indexes/base.py in map(self, mapper, na_action)
       5096         from pandas.core.indexes.multi import MultiIndex
       5097 
    -> 5098         new_values = super()._map_values(mapper, na_action=na_action)
       5099 
       5100         attributes = self._get_attributes_dict()


    ~/opt/anaconda3/lib/python3.8/site-packages/pandas/core/base.py in _map_values(self, mapper, na_action)
        935 
        936         # mapper is a function
    --> 937         new_values = map_f(values, mapper)
        938 
        939         return new_values


    pandas/_libs/lib.pyx in pandas._libs.lib.map_infer()


    <ipython-input-431-50b76488ab6f> in <lambda>(x)
          1 #publish day in the week
    ----> 2 video_df['publishedAt']= video_df['publishedAt'].apply(lambda x: parser.parse(x))
          3 video_df['publishedDayName']= video_df['publishedAt'].apply(lambda x: x.strftime("%A"))


    ~/opt/anaconda3/lib/python3.8/site-packages/dateutil/parser/_parser.py in parse(timestr, parserinfo, **kwargs)
       1372         return parser(parserinfo).parse(timestr, **kwargs)
       1373     else:
    -> 1374         return DEFAULTPARSER.parse(timestr, **kwargs)
       1375 
       1376 


    ~/opt/anaconda3/lib/python3.8/site-packages/dateutil/parser/_parser.py in parse(self, timestr, default, ignoretz, tzinfos, **kwargs)
        644                                                       second=0, microsecond=0)
        645 
    --> 646         res, skipped_tokens = self._parse(timestr, **kwargs)
        647 
        648         if res is None:


    ~/opt/anaconda3/lib/python3.8/site-packages/dateutil/parser/_parser.py in _parse(self, timestr, dayfirst, yearfirst, fuzzy, fuzzy_with_tokens)
        723 
        724         res = self._result()
    --> 725         l = _timelex.split(timestr)         # Splits the timestr into tokens
        726 
        727         skipped_idxs = []


    ~/opt/anaconda3/lib/python3.8/site-packages/dateutil/parser/_parser.py in split(cls, s)
        205     @classmethod
        206     def split(cls, s):
    --> 207         return list(cls(s))
        208 
        209     @classmethod


    ~/opt/anaconda3/lib/python3.8/site-packages/dateutil/parser/_parser.py in __init__(self, instream)
         73             instream = StringIO(instream)
         74         elif getattr(instream, 'read', None) is None:
    ---> 75             raise TypeError('Parser must be a string or character stream, not '
         76                             '{itype}'.format(itype=instream.__class__.__name__))
         77 


    TypeError: Parser must be a string or character stream, not Timestamp



```python
import isodate
video_df['durationSecs'] = video_df['duration'].apply(lambda x: isodate.parse_duration(x))
video_df['durationSecs'] = video_df['durationSecs'].astype('timedelta64[s]')
```


```python
video_df[['durationSecs','duration']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>durationSecs</th>
      <th>duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1747.0</td>
      <td>PT29M7S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1496.0</td>
      <td>PT24M56S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2189.0</td>
      <td>PT36M29S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>127.0</td>
      <td>PT2M7S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3035.0</td>
      <td>PT50M35S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2644.0</td>
      <td>PT44M4S</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3026.0</td>
      <td>PT50M26S</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1848.0</td>
      <td>PT30M48S</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2930.0</td>
      <td>PT48M50S</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1543.0</td>
      <td>PT25M43S</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2866.0</td>
      <td>PT47M46S</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2649.0</td>
      <td>PT44M9S</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3422.0</td>
      <td>PT57M2S</td>
    </tr>
    <tr>
      <th>13</th>
      <td>10151.0</td>
      <td>PT2H49M11S</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.0</td>
      <td>P0D</td>
    </tr>
    <tr>
      <th>15</th>
      <td>46.0</td>
      <td>PT46S</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2382.0</td>
      <td>PT39M42S</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2398.0</td>
      <td>PT39M58S</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2042.0</td>
      <td>PT34M2S</td>
    </tr>
    <tr>
      <th>19</th>
      <td>3401.0</td>
      <td>PT56M41S</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2266.0</td>
      <td>PT37M46S</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2756.0</td>
      <td>PT45M56S</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2465.0</td>
      <td>PT41M5S</td>
    </tr>
    <tr>
      <th>23</th>
      <td>5388.0</td>
      <td>PT1H29M48S</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2294.0</td>
      <td>PT38M14S</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1282.0</td>
      <td>PT21M22S</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1843.0</td>
      <td>PT30M43S</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2914.0</td>
      <td>PT48M34S</td>
    </tr>
    <tr>
      <th>28</th>
      <td>5683.0</td>
      <td>PT1H34M43S</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2082.0</td>
      <td>PT34M42S</td>
    </tr>
    <tr>
      <th>30</th>
      <td>4187.0</td>
      <td>PT1H9M47S</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1525.0</td>
      <td>PT25M25S</td>
    </tr>
    <tr>
      <th>32</th>
      <td>345.0</td>
      <td>PT5M45S</td>
    </tr>
    <tr>
      <th>33</th>
      <td>1588.0</td>
      <td>PT26M28S</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2262.0</td>
      <td>PT37M42S</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1184.0</td>
      <td>PT19M44S</td>
    </tr>
    <tr>
      <th>36</th>
      <td>122.0</td>
      <td>PT2M2S</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2006.0</td>
      <td>PT33M26S</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2806.0</td>
      <td>PT46M46S</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2694.0</td>
      <td>PT44M54S</td>
    </tr>
    <tr>
      <th>40</th>
      <td>3038.0</td>
      <td>PT50M38S</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2385.0</td>
      <td>PT39M45S</td>
    </tr>
    <tr>
      <th>42</th>
      <td>8290.0</td>
      <td>PT2H18M10S</td>
    </tr>
    <tr>
      <th>43</th>
      <td>162.0</td>
      <td>PT2M42S</td>
    </tr>
    <tr>
      <th>44</th>
      <td>201.0</td>
      <td>PT3M21S</td>
    </tr>
    <tr>
      <th>45</th>
      <td>13565.0</td>
      <td>PT3H46M5S</td>
    </tr>
    <tr>
      <th>46</th>
      <td>3029.0</td>
      <td>PT50M29S</td>
    </tr>
    <tr>
      <th>47</th>
      <td>2228.0</td>
      <td>PT37M8S</td>
    </tr>
    <tr>
      <th>48</th>
      <td>1546.0</td>
      <td>PT25M46S</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2050.0</td>
      <td>PT34M10S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Add tag count
video_df['tagCount'] = video_df['tags'].apply(lambda x: 0 if x is None else len(x))
```


```python
video_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>video_id</th>
      <th>channelTitle</th>
      <th>title</th>
      <th>description</th>
      <th>tags</th>
      <th>publishedAt</th>
      <th>viewCount</th>
      <th>likeCount</th>
      <th>favouriteCount</th>
      <th>commentCount</th>
      <th>duration</th>
      <th>definition</th>
      <th>caption</th>
      <th>publishedDayName</th>
      <th>durationSecs</th>
      <th>tagCount</th>
      <th>title_no_stopwords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>VN8pnrs1jBE</td>
      <td>Markiplier</td>
      <td>IT'S ENDING ME!! | Endoparasitic - Part 3</td>
      <td>This is the ending of Endoparasitic! I hope we...</td>
      <td>None</td>
      <td>2023-01-03 17:00:14+00:00</td>
      <td>1201399.0</td>
      <td>70034.0</td>
      <td>NaN</td>
      <td>3786.0</td>
      <td>PT29M7S</td>
      <td>hd</td>
      <td>false</td>
      <td>Tuesday</td>
      <td>1747.0</td>
      <td>0</td>
      <td>[IT'S, ENDING, ME!!, |, Endoparasitic, -, Part...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0ALfWW_U728</td>
      <td>Markiplier</td>
      <td>My Mom Plays Five Nights at Freddy's: Sister L...</td>
      <td>Markiplier's Mom attempts to beat Five Nights ...</td>
      <td>[markiplier, fnaf, five nights at freddy's, fi...</td>
      <td>2023-01-02 17:35:18+00:00</td>
      <td>1998009.0</td>
      <td>192801.0</td>
      <td>NaN</td>
      <td>12319.0</td>
      <td>PT24M56S</td>
      <td>hd</td>
      <td>false</td>
      <td>Monday</td>
      <td>1496.0</td>
      <td>22</td>
      <td>[My, Mom, Plays, Five, Nights, Freddy's:, Sist...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>W2NafoExi1Y</td>
      <td>Markiplier</td>
      <td>best raft ever… | Raft</td>
      <td>This is the best raft we've ever built. ever. ...</td>
      <td>[markiplier, raft, muyskerm, lordminion777, ma...</td>
      <td>2022-12-30 17:00:30+00:00</td>
      <td>1810070.0</td>
      <td>82449.0</td>
      <td>NaN</td>
      <td>4119.0</td>
      <td>PT36M29S</td>
      <td>hd</td>
      <td>false</td>
      <td>Friday</td>
      <td>2189.0</td>
      <td>8</td>
      <td>[best, raft, ever…, |, Raft]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5DwIJPSPtFg</td>
      <td>Markiplier</td>
      <td>The Fall of Markiplier</td>
      <td>Mark takes a tumble. Here's the story of the p...</td>
      <td>[markiplier, animated, distractible, mark bob ...</td>
      <td>2022-12-28 21:25:36+00:00</td>
      <td>1728446.0</td>
      <td>162281.0</td>
      <td>NaN</td>
      <td>5483.0</td>
      <td>PT2M7S</td>
      <td>hd</td>
      <td>false</td>
      <td>Wednesday</td>
      <td>127.0</td>
      <td>20</td>
      <td>[The, Fall, Markiplier]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>299Zd7nmqdg</td>
      <td>Markiplier</td>
      <td>...OOPS! | Shadows Over Loathing - Part 4</td>
      <td>Ah... so that's... that's not gonna come back,...</td>
      <td>[shadows over loathing, west of loathing, funn...</td>
      <td>2022-12-27 17:00:01+00:00</td>
      <td>1154311.0</td>
      <td>58126.0</td>
      <td>NaN</td>
      <td>3016.0</td>
      <td>PT50M35S</td>
      <td>hd</td>
      <td>false</td>
      <td>Tuesday</td>
      <td>3035.0</td>
      <td>12</td>
      <td>[...OOPS!, |, Shadows, Over, Loathing, -, Part...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>je_HfFoHq6g</td>
      <td>Markiplier</td>
      <td>Bendy and the Dark Revival: Part 3</td>
      <td>The citizens of Bendy and the Dark Revival are...</td>
      <td>[bendy and the dark revival, bendy and the ink...</td>
      <td>2022-12-26 17:00:32+00:00</td>
      <td>1115577.0</td>
      <td>60028.0</td>
      <td>NaN</td>
      <td>3828.0</td>
      <td>PT44M4S</td>
      <td>hd</td>
      <td>false</td>
      <td>Monday</td>
      <td>2644.0</td>
      <td>15</td>
      <td>[Bendy, Dark, Revival:, Part, 3]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>L0LAE2FFrfA</td>
      <td>Markiplier</td>
      <td>IT'S OUTSIDE ME!! | Endoparasitic - Part 2</td>
      <td>What if the monster that WAS inside of you sud...</td>
      <td>[markiplier, scary games, endoparasitic, horro...</td>
      <td>2022-12-25 17:37:47+00:00</td>
      <td>2038489.0</td>
      <td>91922.0</td>
      <td>NaN</td>
      <td>4429.0</td>
      <td>PT50M26S</td>
      <td>hd</td>
      <td>false</td>
      <td>Sunday</td>
      <td>3026.0</td>
      <td>4</td>
      <td>[IT'S, OUTSIDE, ME!!, |, Endoparasitic, -, Par...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>byaGmRCDmtY</td>
      <td>Markiplier</td>
      <td>birds… | Raft</td>
      <td>The bird… it bothers us… \nLISTEN TO DISTRACTI...</td>
      <td>None</td>
      <td>2022-12-22 19:01:20+00:00</td>
      <td>1671340.0</td>
      <td>81716.0</td>
      <td>NaN</td>
      <td>5775.0</td>
      <td>PT30M48S</td>
      <td>hd</td>
      <td>false</td>
      <td>Thursday</td>
      <td>1848.0</td>
      <td>0</td>
      <td>[birds…, |, Raft]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>FRUTDiguCgw</td>
      <td>Markiplier</td>
      <td>NOT THE TENTACLES!! | The Callisto Protocol - ...</td>
      <td>Turns out the monsters from Callisto Protocol ...</td>
      <td>[markiplier, callisto protocol, dead space, pa...</td>
      <td>2022-12-19 17:00:07+00:00</td>
      <td>1410505.0</td>
      <td>62698.0</td>
      <td>NaN</td>
      <td>4116.0</td>
      <td>PT48M50S</td>
      <td>hd</td>
      <td>false</td>
      <td>Monday</td>
      <td>2930.0</td>
      <td>10</td>
      <td>[NOT, THE, TENTACLES!!, |, The, Callisto, Prot...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>WScLAdh6dlE</td>
      <td>Markiplier</td>
      <td>hanging by a thread... | Raft</td>
      <td>Supplies are dwindling... we'll need to find s...</td>
      <td>[markiplier, raft, muyskerm, lordminion777, ma...</td>
      <td>2022-12-16 18:45:34+00:00</td>
      <td>1756819.0</td>
      <td>84267.0</td>
      <td>NaN</td>
      <td>4032.0</td>
      <td>PT25M43S</td>
      <td>hd</td>
      <td>false</td>
      <td>Friday</td>
      <td>1543.0</td>
      <td>8</td>
      <td>[hanging, thread..., |, Raft]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>AtxYF4mXK7w</td>
      <td>Markiplier</td>
      <td>IT'S INSIDE ME!! | Endoparasitic</td>
      <td>What if monsters ripped off all your limbs exc...</td>
      <td>[markiplier, scary games, endoparasitic, horro...</td>
      <td>2022-12-14 18:14:48+00:00</td>
      <td>4539525.0</td>
      <td>203000.0</td>
      <td>NaN</td>
      <td>7862.0</td>
      <td>PT47M46S</td>
      <td>hd</td>
      <td>false</td>
      <td>Wednesday</td>
      <td>2866.0</td>
      <td>4</td>
      <td>[IT'S, INSIDE, ME!!, |, Endoparasitic]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3ZlKLi0Tz3I</td>
      <td>Markiplier</td>
      <td>I'M... A DINOSAUR? | Shadows Over Loathing - P...</td>
      <td>Well... Shadows Over Loathing has taken a surp...</td>
      <td>[shadows over loathing, west of loathing, funn...</td>
      <td>2022-12-13 21:57:17+00:00</td>
      <td>1494692.0</td>
      <td>73714.0</td>
      <td>NaN</td>
      <td>3501.0</td>
      <td>PT44M9S</td>
      <td>hd</td>
      <td>false</td>
      <td>Tuesday</td>
      <td>2649.0</td>
      <td>12</td>
      <td>[I'M..., A, DINOSAUR?, |, Shadows, Over, Loath...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-x4OkAaaCaU</td>
      <td>Markiplier</td>
      <td>we're DEAD and in SPACE!! | The Callisto Protocol</td>
      <td>From some of the developers of Dead Space come...</td>
      <td>[markiplier, callisto protocol, dead space, pa...</td>
      <td>2022-12-12 18:57:20+00:00</td>
      <td>2185814.0</td>
      <td>111649.0</td>
      <td>NaN</td>
      <td>6676.0</td>
      <td>PT57M2S</td>
      <td>hd</td>
      <td>false</td>
      <td>Monday</td>
      <td>3422.0</td>
      <td>10</td>
      <td>[we're, DEAD, SPACE!!, |, The, Callisto, Proto...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>seD3OV70KH8</td>
      <td>Markiplier</td>
      <td>Choo Choo Charles</td>
      <td>I'm the ONLY CEO ►► https://cloakbrand.com/</td>
      <td>[markiplier, choo choo charles]</td>
      <td>2022-12-11 23:12:49+00:00</td>
      <td>9492987.0</td>
      <td>262551.0</td>
      <td>NaN</td>
      <td>9830.0</td>
      <td>PT2H49M11S</td>
      <td>hd</td>
      <td>false</td>
      <td>Sunday</td>
      <td>10151.0</td>
      <td>2</td>
      <td>[Choo, Choo, Charles]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>60hBe24JXlc</td>
      <td>Markiplier</td>
      <td>Choo Choo Charles</td>
      <td>I'm the ONLY CEO ►► https://cloakbrand.com/</td>
      <td>[markiplier, choo choo charles]</td>
      <td>2022-12-11 22:58:08+00:00</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>P0D</td>
      <td>sd</td>
      <td>false</td>
      <td>Sunday</td>
      <td>0.0</td>
      <td>2</td>
      <td>[Choo, Choo, Charles]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-plcKTTkhaI</td>
      <td>Markiplier</td>
      <td>here...</td>
      <td>Markiplier's Official Only Fans ► https://only...</td>
      <td>[markiplier, only fans]</td>
      <td>2022-12-09 02:21:00+00:00</td>
      <td>4203759.0</td>
      <td>331807.0</td>
      <td>NaN</td>
      <td>21653.0</td>
      <td>PT46S</td>
      <td>hd</td>
      <td>false</td>
      <td>Friday</td>
      <td>46.0</td>
      <td>2</td>
      <td>[here...]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>LugetKGwa1A</td>
      <td>Markiplier</td>
      <td>Bendy and the Dark Revival: Part 2</td>
      <td>Madness? In this place? You've gotta be crazy ...</td>
      <td>[bendy and the dark revival, bendy and the ink...</td>
      <td>2022-12-07 21:47:18+00:00</td>
      <td>1688450.0</td>
      <td>80477.0</td>
      <td>NaN</td>
      <td>3947.0</td>
      <td>PT39M42S</td>
      <td>hd</td>
      <td>false</td>
      <td>Wednesday</td>
      <td>2382.0</td>
      <td>15</td>
      <td>[Bendy, Dark, Revival:, Part, 2]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>-SA4dOgLePk</td>
      <td>Markiplier</td>
      <td>OH, NEAT! I'M HAUNTED! | Shadows Over Loathing...</td>
      <td>We continue our adventure into Shadows of Loat...</td>
      <td>[shadows over loathing, west of loathing, funn...</td>
      <td>2022-12-06 23:54:24+00:00</td>
      <td>2037812.0</td>
      <td>96357.0</td>
      <td>NaN</td>
      <td>4661.0</td>
      <td>PT39M58S</td>
      <td>hd</td>
      <td>false</td>
      <td>Tuesday</td>
      <td>2398.0</td>
      <td>12</td>
      <td>[OH,, NEAT!, I'M, HAUNTED!, |, Shadows, Over, ...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>XTR8NvITdxo</td>
      <td>Markiplier</td>
      <td>drowning... | Raft</td>
      <td>Don't mind me, just drowning a bit...\nLISTEN ...</td>
      <td>[markiplier, raft, muyskerm, lordminion777, ma...</td>
      <td>2022-12-05 17:00:14+00:00</td>
      <td>1870876.0</td>
      <td>89256.0</td>
      <td>NaN</td>
      <td>3465.0</td>
      <td>PT34M2S</td>
      <td>hd</td>
      <td>false</td>
      <td>Monday</td>
      <td>2042.0</td>
      <td>8</td>
      <td>[drowning..., |, Raft]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>lGyBzNxZWms</td>
      <td>Markiplier</td>
      <td>Bendy and the Dark Revival: Part 1</td>
      <td>We return to the wonderful and/or horrible Dis...</td>
      <td>[bendy and the dark revival, bendy and the ink...</td>
      <td>2022-12-04 20:28:08+00:00</td>
      <td>3235395.0</td>
      <td>151589.0</td>
      <td>NaN</td>
      <td>6532.0</td>
      <td>PT56M41S</td>
      <td>hd</td>
      <td>false</td>
      <td>Sunday</td>
      <td>3401.0</td>
      <td>15</td>
      <td>[Bendy, Dark, Revival:, Part, 1]</td>
    </tr>
    <tr>
      <th>20</th>
      <td>xhuqEPMUzOw</td>
      <td>Markiplier</td>
      <td>island of death... | Raft</td>
      <td>There's horrible things on this island... horr...</td>
      <td>[markiplier, raft, muyskerm, lordminion777, ma...</td>
      <td>2022-12-02 18:21:57+00:00</td>
      <td>1961643.0</td>
      <td>83200.0</td>
      <td>NaN</td>
      <td>2927.0</td>
      <td>PT37M46S</td>
      <td>hd</td>
      <td>false</td>
      <td>Friday</td>
      <td>2266.0</td>
      <td>8</td>
      <td>[island, death..., |, Raft]</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1Ivc8WHfhnY</td>
      <td>Markiplier</td>
      <td>CTHULHU, BUT FUNNY | Shadows Over Loathing - P...</td>
      <td>Shadows Over Loathing, the sequel to West of L...</td>
      <td>[shadows over loathing, west of loathing, funn...</td>
      <td>2022-12-01 23:34:40+00:00</td>
      <td>3043532.0</td>
      <td>161157.0</td>
      <td>NaN</td>
      <td>15834.0</td>
      <td>PT45M56S</td>
      <td>hd</td>
      <td>false</td>
      <td>Thursday</td>
      <td>2756.0</td>
      <td>12</td>
      <td>[CTHULHU,, BUT, FUNNY, |, Shadows, Over, Loath...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>lD3s3jsw3pc</td>
      <td>Markiplier</td>
      <td>we found a message... | Raft</td>
      <td>There's a message for us...\nLISTEN TO DISTRAC...</td>
      <td>[markiplier, raft, muyskerm, lordminion777, ma...</td>
      <td>2022-11-30 18:57:08+00:00</td>
      <td>2025223.0</td>
      <td>85940.0</td>
      <td>NaN</td>
      <td>3089.0</td>
      <td>PT41M5S</td>
      <td>hd</td>
      <td>false</td>
      <td>Wednesday</td>
      <td>2465.0</td>
      <td>8</td>
      <td>[found, message..., |, Raft]</td>
    </tr>
    <tr>
      <th>23</th>
      <td>KNq35Z8a_QU</td>
      <td>Markiplier</td>
      <td>FAITH: Chapter 3</td>
      <td>FAITH makes a long-awaited return in THE UNHOL...</td>
      <td>[faith the unholy trinity, markiplier, faith c...</td>
      <td>2022-11-28 19:53:13+00:00</td>
      <td>2366361.0</td>
      <td>79640.0</td>
      <td>NaN</td>
      <td>6493.0</td>
      <td>PT1H29M48S</td>
      <td>hd</td>
      <td>false</td>
      <td>Monday</td>
      <td>5388.0</td>
      <td>16</td>
      <td>[FAITH:, Chapter, 3]</td>
    </tr>
    <tr>
      <th>24</th>
      <td>GhYOdwLVBYs</td>
      <td>Markiplier</td>
      <td>its a boat... | Raft</td>
      <td>The boys and I make some more progress and dis...</td>
      <td>[markiplier, raft, muyskerm, lordminion777, ma...</td>
      <td>2022-11-27 22:45:08+00:00</td>
      <td>2126170.0</td>
      <td>102494.0</td>
      <td>NaN</td>
      <td>4323.0</td>
      <td>PT38M14S</td>
      <td>hd</td>
      <td>false</td>
      <td>Sunday</td>
      <td>2294.0</td>
      <td>8</td>
      <td>[boat..., |, Raft]</td>
    </tr>
    <tr>
      <th>25</th>
      <td>w9WhPHLTdYc</td>
      <td>Markiplier</td>
      <td>THE LADY RETURNS | Resident Evil: Village DLC ...</td>
      <td>I can hardly believe it... after all this time...</td>
      <td>[markiplier, lady dimitrescu, resident evil vi...</td>
      <td>2022-11-25 18:57:27+00:00</td>
      <td>1189709.0</td>
      <td>78906.0</td>
      <td>NaN</td>
      <td>4294.0</td>
      <td>PT21M22S</td>
      <td>hd</td>
      <td>false</td>
      <td>Friday</td>
      <td>1282.0</td>
      <td>16</td>
      <td>[THE, LADY, RETURNS, |, Resident, Evil:, Villa...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>iOjwpYTSiVk</td>
      <td>Markiplier</td>
      <td>END OF ROSE | Resident Evil: Village DLC - Part 5</td>
      <td>We reach the end of Shadow of Rose... but we s...</td>
      <td>[markiplier, lady dimitrescu, resident evil vi...</td>
      <td>2022-11-23 19:58:26+00:00</td>
      <td>1074627.0</td>
      <td>67378.0</td>
      <td>NaN</td>
      <td>3430.0</td>
      <td>PT30M43S</td>
      <td>hd</td>
      <td>false</td>
      <td>Wednesday</td>
      <td>1843.0</td>
      <td>13</td>
      <td>[END, OF, ROSE, |, Resident, Evil:, Village, D...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>MISAiE64vD0</td>
      <td>Markiplier</td>
      <td>SPIDER HELL | Grounded - Part 5</td>
      <td>We find out the hard way that spiders in Groun...</td>
      <td>[grounded, grounded game, grounded markiplier,...</td>
      <td>2022-11-22 17:47:08+00:00</td>
      <td>2043237.0</td>
      <td>81239.0</td>
      <td>NaN</td>
      <td>3146.0</td>
      <td>PT48M34S</td>
      <td>hd</td>
      <td>false</td>
      <td>Tuesday</td>
      <td>2914.0</td>
      <td>18</td>
      <td>[SPIDER, HELL, |, Grounded, -, Part, 5]</td>
    </tr>
    <tr>
      <th>28</th>
      <td>F6Wc11ErqmA</td>
      <td>Markiplier</td>
      <td>Getting Over It Again</td>
      <td>Mystery Box ►► https://cloakbrand.com/</td>
      <td>[getting over it]</td>
      <td>2022-11-21 21:00:05+00:00</td>
      <td>3294643.0</td>
      <td>143083.0</td>
      <td>NaN</td>
      <td>5450.0</td>
      <td>PT1H34M43S</td>
      <td>hd</td>
      <td>false</td>
      <td>Monday</td>
      <td>5683.0</td>
      <td>1</td>
      <td>[Getting, Over, It, Again]</td>
    </tr>
    <tr>
      <th>29</th>
      <td>vcm2Eccz7EE</td>
      <td>Markiplier</td>
      <td>EVIL RETURNS!! | Resident Evil: Village DLC - ...</td>
      <td>Oh hi, Eveline! Long time no see!\nLast Chance...</td>
      <td>[markiplier, lady dimitrescu, resident evil vi...</td>
      <td>2022-11-20 19:19:30+00:00</td>
      <td>1309870.0</td>
      <td>81084.0</td>
      <td>NaN</td>
      <td>3347.0</td>
      <td>PT34M42S</td>
      <td>hd</td>
      <td>false</td>
      <td>Sunday</td>
      <td>2082.0</td>
      <td>13</td>
      <td>[EVIL, RETURNS!!, |, Resident, Evil:, Village,...</td>
    </tr>
    <tr>
      <th>30</th>
      <td>63h6RH5u984</td>
      <td>Markiplier</td>
      <td>there is time</td>
      <td>Last Chance ►► https://www.moment.co/markiplier</td>
      <td>[markiplier]</td>
      <td>2022-11-20 02:46:32+00:00</td>
      <td>1301048.0</td>
      <td>60898.0</td>
      <td>NaN</td>
      <td>3764.0</td>
      <td>PT1H9M47S</td>
      <td>hd</td>
      <td>false</td>
      <td>Sunday</td>
      <td>4187.0</td>
      <td>1</td>
      <td>[time]</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Lvicaufi5O0</td>
      <td>Markiplier</td>
      <td>Mori</td>
      <td>Memento ► https://youtu.be/yGxOsZOyyRg</td>
      <td>[memento mori, unus annus]</td>
      <td>2022-11-13 20:02:00+00:00</td>
      <td>3635816.0</td>
      <td>298099.0</td>
      <td>NaN</td>
      <td>20965.0</td>
      <td>PT25M25S</td>
      <td>hd</td>
      <td>false</td>
      <td>Sunday</td>
      <td>1525.0</td>
      <td>2</td>
      <td>[Mori]</td>
    </tr>
    <tr>
      <th>32</th>
      <td>DATcijR4uNM</td>
      <td>Markiplier</td>
      <td>there's no time</td>
      <td>BE HERE ON SUNDAY ►► https://www.moment.co/mar...</td>
      <td>None</td>
      <td>2022-11-11 18:52:42+00:00</td>
      <td>1759102.0</td>
      <td>155571.0</td>
      <td>NaN</td>
      <td>6863.0</td>
      <td>PT5M45S</td>
      <td>hd</td>
      <td>false</td>
      <td>Friday</td>
      <td>345.0</td>
      <td>0</td>
      <td>[there's, time]</td>
    </tr>
    <tr>
      <th>33</th>
      <td>vPTyH7Bx2Fc</td>
      <td>Markiplier</td>
      <td>FNAF Escape Room | The Glitched Attraction</td>
      <td>King of FNAF YouTooz ►► https://youtooz.com/pr...</td>
      <td>[the glitched attraction, five nights at fredd...</td>
      <td>2022-11-06 22:08:15+00:00</td>
      <td>4693085.0</td>
      <td>256144.0</td>
      <td>NaN</td>
      <td>10664.0</td>
      <td>PT26M28S</td>
      <td>hd</td>
      <td>false</td>
      <td>Sunday</td>
      <td>1588.0</td>
      <td>5</td>
      <td>[FNAF, Escape, Room, |, The, Glitched, Attract...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>572akbCNfRc</td>
      <td>Markiplier</td>
      <td>NOT THE MANNEQUINS... | Resident Evil: Village...</td>
      <td>Mannequins... why does it always have to be ma...</td>
      <td>[markiplier, lady dimitrescu, resident evil vi...</td>
      <td>2022-11-05 18:41:54+00:00</td>
      <td>2099893.0</td>
      <td>113525.0</td>
      <td>NaN</td>
      <td>5807.0</td>
      <td>PT37M42S</td>
      <td>hd</td>
      <td>false</td>
      <td>Saturday</td>
      <td>2262.0</td>
      <td>13</td>
      <td>[NOT, THE, MANNEQUINS..., |, Resident, Evil:, ...</td>
    </tr>
    <tr>
      <th>35</th>
      <td>7MysgX9vv48</td>
      <td>Markiplier</td>
      <td>Amanda the Adventurer: UPDATE</td>
      <td>Amanda the Adventurer has one more adventure t...</td>
      <td>None</td>
      <td>2022-11-03 18:28:39+00:00</td>
      <td>2423337.0</td>
      <td>143228.0</td>
      <td>NaN</td>
      <td>5795.0</td>
      <td>PT19M44S</td>
      <td>hd</td>
      <td>false</td>
      <td>Thursday</td>
      <td>1184.0</td>
      <td>0</td>
      <td>[Amanda, Adventurer:, UPDATE]</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Mij7pTfuex4</td>
      <td>Markiplier</td>
      <td>you win...</td>
      <td>You didn't forget... did you? ► https://www.mo...</td>
      <td>[markiplier]</td>
      <td>2022-11-02 21:11:00+00:00</td>
      <td>4872206.0</td>
      <td>488530.0</td>
      <td>NaN</td>
      <td>20918.0</td>
      <td>PT2M2S</td>
      <td>hd</td>
      <td>false</td>
      <td>Wednesday</td>
      <td>122.0</td>
      <td>1</td>
      <td>[win...]</td>
    </tr>
    <tr>
      <th>37</th>
      <td>7GbDedt1nkw</td>
      <td>Markiplier</td>
      <td>WHERE IS SHE?! | Resident Evil: Village DLC - ...</td>
      <td>We continue to explore the manor of Lady Dimit...</td>
      <td>[markiplier, lady dimitrescu, resident evil vi...</td>
      <td>2022-11-01 16:18:11+00:00</td>
      <td>2131346.0</td>
      <td>114195.0</td>
      <td>NaN</td>
      <td>3460.0</td>
      <td>PT33M26S</td>
      <td>hd</td>
      <td>false</td>
      <td>Tuesday</td>
      <td>2006.0</td>
      <td>12</td>
      <td>[WHERE, IS, SHE?!, |, Resident, Evil:, Village...</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1VOa-7HmZ0M</td>
      <td>Markiplier</td>
      <td>just a dream... | Raft</td>
      <td>I really hope this is all just a dream...\nGet...</td>
      <td>[markiplier, raft, muyskerm, lordminion777, ma...</td>
      <td>2022-10-31 16:05:40+00:00</td>
      <td>3121253.0</td>
      <td>117552.0</td>
      <td>NaN</td>
      <td>3877.0</td>
      <td>PT46M46S</td>
      <td>hd</td>
      <td>false</td>
      <td>Monday</td>
      <td>2806.0</td>
      <td>8</td>
      <td>[dream..., |, Raft]</td>
    </tr>
    <tr>
      <th>39</th>
      <td>JRswsF1SLok</td>
      <td>Markiplier</td>
      <td>SHADOW OF ROSE | Resident Evil: Village DLC - ...</td>
      <td>Resident Evil: Village has finally released it...</td>
      <td>[markiplier, lady dimitrescu, resident evil vi...</td>
      <td>2022-10-30 16:53:14+00:00</td>
      <td>3864345.0</td>
      <td>202882.0</td>
      <td>NaN</td>
      <td>8254.0</td>
      <td>PT44M54S</td>
      <td>hd</td>
      <td>false</td>
      <td>Sunday</td>
      <td>2694.0</td>
      <td>12</td>
      <td>[SHADOW, OF, ROSE, |, Resident, Evil:, Village...</td>
    </tr>
    <tr>
      <th>40</th>
      <td>S_1LnTo781Q</td>
      <td>Markiplier</td>
      <td>MARKIPLIER IS TINY (again...) | Grounded - Part 4</td>
      <td>Grounded has been fully released which means i...</td>
      <td>[grounded, grounded game, grounded markiplier,...</td>
      <td>2022-10-29 19:37:33+00:00</td>
      <td>2492362.0</td>
      <td>98887.0</td>
      <td>NaN</td>
      <td>4755.0</td>
      <td>PT50M38S</td>
      <td>hd</td>
      <td>false</td>
      <td>Saturday</td>
      <td>3038.0</td>
      <td>18</td>
      <td>[MARKIPLIER, IS, TINY, (again...), |, Grounded...</td>
    </tr>
    <tr>
      <th>41</th>
      <td>pNhzuC647gA</td>
      <td>Markiplier</td>
      <td>3 SCARY GAMES #95</td>
      <td>Melissa met The Man in the Park on Halloween. ...</td>
      <td>[markiplier, 3 scary games 95, melissa, the ma...</td>
      <td>2022-10-27 17:07:04+00:00</td>
      <td>3572607.0</td>
      <td>165255.0</td>
      <td>NaN</td>
      <td>9870.0</td>
      <td>PT39M45S</td>
      <td>hd</td>
      <td>false</td>
      <td>Thursday</td>
      <td>2385.0</td>
      <td>9</td>
      <td>[3, SCARY, GAMES, #95]</td>
    </tr>
    <tr>
      <th>42</th>
      <td>cxbxfC-3d2A</td>
      <td>Markiplier</td>
      <td>fine</td>
      <td>My Mom's Documentary ►► https://www.moment.co/...</td>
      <td>None</td>
      <td>2022-10-26 23:09:08+00:00</td>
      <td>4451424.0</td>
      <td>224781.0</td>
      <td>NaN</td>
      <td>9333.0</td>
      <td>PT2H18M10S</td>
      <td>hd</td>
      <td>false</td>
      <td>Wednesday</td>
      <td>8290.0</td>
      <td>0</td>
      <td>[fine]</td>
    </tr>
    <tr>
      <th>43</th>
      <td>5ZcU1djtd1M</td>
      <td>Markiplier</td>
      <td>...wow</td>
      <td>Don't Disappoint My Mom ► https://www.moment.c...</td>
      <td>[markiplier]</td>
      <td>2022-10-22 01:07:38+00:00</td>
      <td>6082383.0</td>
      <td>709333.0</td>
      <td>NaN</td>
      <td>34477.0</td>
      <td>PT2M42S</td>
      <td>hd</td>
      <td>false</td>
      <td>Saturday</td>
      <td>162.0</td>
      <td>1</td>
      <td>[...wow]</td>
    </tr>
    <tr>
      <th>44</th>
      <td>mOgQDKOGZ1k</td>
      <td>Markiplier</td>
      <td>I Will Start an Only Fans...</td>
      <td>Distractible Apple ► https://podcasts.apple.co...</td>
      <td>[markiplier]</td>
      <td>2022-10-16 22:17:29+00:00</td>
      <td>7146583.0</td>
      <td>826003.0</td>
      <td>NaN</td>
      <td>52228.0</td>
      <td>PT3M21S</td>
      <td>hd</td>
      <td>false</td>
      <td>Sunday</td>
      <td>201.0</td>
      <td>1</td>
      <td>[I, Will, Start, Only, Fans...]</td>
    </tr>
    <tr>
      <th>45</th>
      <td>wIp7dqfICxI</td>
      <td>Markiplier</td>
      <td>we haven't seen each other in 5 years</td>
      <td>CHECK THIS OUT  ►► https://open.spotify.com/sh...</td>
      <td>[markiplier, muyskerm, lordminion777, mark bob...</td>
      <td>2022-10-15 00:32:17+00:00</td>
      <td>4394159.0</td>
      <td>145777.0</td>
      <td>NaN</td>
      <td>3760.0</td>
      <td>PT3H46M5S</td>
      <td>hd</td>
      <td>false</td>
      <td>Saturday</td>
      <td>13565.0</td>
      <td>5</td>
      <td>[seen, 5, years]</td>
    </tr>
    <tr>
      <th>46</th>
      <td>XC3P-eZ3a04</td>
      <td>Markiplier</td>
      <td>Broken Through</td>
      <td>The Wonderful World of Distractible ►► https:/...</td>
      <td>None</td>
      <td>2022-10-10 16:56:02+00:00</td>
      <td>3526209.0</td>
      <td>144433.0</td>
      <td>NaN</td>
      <td>10057.0</td>
      <td>PT50M29S</td>
      <td>hd</td>
      <td>false</td>
      <td>Monday</td>
      <td>3029.0</td>
      <td>0</td>
      <td>[Broken, Through]</td>
    </tr>
    <tr>
      <th>47</th>
      <td>5ImydqxxGDM</td>
      <td>Markiplier</td>
      <td>im on shark duty... | Raft</td>
      <td>CLICK HERE ►► https://open.spotify.com/episode...</td>
      <td>None</td>
      <td>2022-10-08 16:00:02+00:00</td>
      <td>2868917.0</td>
      <td>120191.0</td>
      <td>NaN</td>
      <td>5102.0</td>
      <td>PT37M8S</td>
      <td>hd</td>
      <td>false</td>
      <td>Saturday</td>
      <td>2228.0</td>
      <td>0</td>
      <td>[im, shark, duty..., |, Raft]</td>
    </tr>
    <tr>
      <th>48</th>
      <td>orXCm-k8c94</td>
      <td>Markiplier</td>
      <td>3 SCARY GAMES #94</td>
      <td>Spongebob is coming for you!! No, this isn't a...</td>
      <td>[markiplier, 3 scary games 94, the true ingred...</td>
      <td>2022-10-07 16:00:39+00:00</td>
      <td>4365637.0</td>
      <td>211629.0</td>
      <td>NaN</td>
      <td>21370.0</td>
      <td>PT25M46S</td>
      <td>hd</td>
      <td>false</td>
      <td>Friday</td>
      <td>1546.0</td>
      <td>11</td>
      <td>[3, SCARY, GAMES, #94]</td>
    </tr>
    <tr>
      <th>49</th>
      <td>YNykdHDENIY</td>
      <td>Markiplier</td>
      <td>a little too comfortable... | Raft</td>
      <td>Things are going well... too well...\nTHE LINK...</td>
      <td>[markiplier, raft, muyskerm, lordminion777, ma...</td>
      <td>2022-10-06 18:55:28+00:00</td>
      <td>2866726.0</td>
      <td>128531.0</td>
      <td>NaN</td>
      <td>4710.0</td>
      <td>PT34M10S</td>
      <td>hd</td>
      <td>false</td>
      <td>Thursday</td>
      <td>2050.0</td>
      <td>8</td>
      <td>[little, comfortable..., |, Raft]</td>
    </tr>
  </tbody>
</table>
</div>



# EDA

## Best performing videos


```python
ax = sns.barplot(x = 'title', y = 'viewCount', data = video_df.sort_values('viewCount', ascending=False)[0:9])
plot = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos:'{:,.0f}'.format(x/1000) + 'K'))
```


    
![png](output_23_0.png)
    



```python
ax = sns.barplot(x = 'title', y = 'viewCount', data = video_df.sort_values('viewCount', ascending=True)[0:9])
plot = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos:'{:,.0f}'.format(x/1000) + 'K'))
```


    
![png](output_24_0.png)
    


## View distribution per video


```python
sns.violinplot(video_df['channelTitle'], video_df['viewCount'])
```

    /Users/apple/opt/anaconda3/lib/python3.8/site-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(





    <AxesSubplot:xlabel='channelTitle', ylabel='viewCount'>




    
![png](output_26_2.png)
    


## Views vs. likes and comments



```python
fig, ax = plt.subplots(1,2)
sns.scatterplot(data = video_df, x = 'commentCount', y = 'viewCount', ax = ax[0])
sns.scatterplot(data = video_df, x = 'likeCount', y = 'viewCount', ax = ax[1])

```




    <AxesSubplot:xlabel='likeCount', ylabel='viewCount'>




    
![png](output_28_1.png)
    


## Video duration


```python
sns.histplot(data = video_df, x = 'durationSecs', bins=10)
```




    <AxesSubplot:xlabel='durationSecs', ylabel='Count'>




    
![png](output_30_1.png)
    


## Wordcloud for video titles



```python
stop_words = set(stopwords.words('english'))
video_df['title_no_stopwords'] = video_df['title'].apply(lambda x: [item for item in str(x).split() if item not in stop_words])

all_words = list([a for b in video_df['title_no_stopwords'].tolist() for a in b])
all_words_str = ' '.join(all_words) 

def plot_cloud(wordcloud):
    plt.figure(figsize=(30, 20))
    plt.imshow(wordcloud) 
    plt.axis("off");

wordcloud = WordCloud(width = 2000, height = 1000, random_state=1, background_color='black', 
                      colormap='viridis', collocations=False).generate(all_words_str)
plot_cloud(wordcloud)
```


    
![png](output_32_0.png)
    


## upload schedule


```python
day_df = pd.DataFrame(video_df['publishedDayName'].value_counts())
weekdays = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_df = day_df.reindex(weekdays)
ax = day_df.reset_index().plot.bar(x='index', y='publishedDayName', rot=0)
```


    
![png](output_34_0.png)
    



```python

```


```python

```


```python

```
