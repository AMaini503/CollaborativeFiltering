
# coding: utf-8

# In[1]:


sc


# In[2]:


rawUserArtistData = sc.textFile('audio_data/user_artist_data.txt')


# In[3]:


rawUserArtistData.getNumPartitions()


# In[41]:


userIDs = rawUserArtistData.map(lambda line: float(line.split()[0]))


# In[5]:


artistIDs = rawUserArtistData.map(lambda line: float(line.split()[1]))


# ### Maximum value matches as given in the book

# In[6]:


userIDs.stats()


# In[7]:


artistIDs.stats()


# ### Construct ID to Artist Name Map

# In[3]:


rawArtistData = sc.textFile('audio_data/artist_data.txt')


# In[4]:


def getArtistIDAndName(line):
    """Gets artist id and name from a line in artist_data.txt"""
    
    # split about tab
    tokens = line.split('\t')
    
    try:
        return [int(tokens[0]), tokens[1].strip()]
    except Exception as e:
        return None


# In[45]:


id2name = rawArtistData.map(lambda line: getArtistIDAndName(line))


# In[32]:


# id2name.count()


# In[46]:


id2name.take(5)


# #### Remove None elements for invalid lines

# In[62]:


id2name = id2name.filter(lambda ele: ele is not None)


# #### No invalid elements

# In[35]:


# none_count = id2name.filter(lambda ele: ele is None)


# In[36]:


# none_count.count()


# #### Sanity Check

# In[16]:


# id2name.lookup(1134999)


# #### Construct Alias to Canonical Name mapping

# In[5]:


rawArtistAlias = sc.textFile('audio_data/artist_alias.txt')


# In[6]:


def MapAliasToCanonical(line):
    tokens = line.split('\t')
    
    try: 
        return [int(tokens[0]), int(tokens[1])]
    except Exception as e:
        return None


# In[7]:


artistAlias = rawArtistAlias.map(lambda line: MapAliasToCanonical(line))


# #### Remove None elements

# In[8]:


artistAlias = artistAlias.filter(lambda ele: ele is not None)


# In[21]:


# artistAlias.count()


# In[22]:


# artistAlias.take(10)


# #### No invalid elements

# In[23]:


# none_count = artistAlias.filter(lambda ele: ele is None)


# In[24]:


# none_count.count()


# #### Sanity Check

# In[25]:


# print(artistAlias.lookup(6803336))


# In[37]:


# del id2name, userIDs, artistIDs


# ### Building the Model

# In[9]:


from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating


# In[10]:


bArtistAlias = sc.broadcast(artistAlias.collectAsMap())


# #### Sanity Check the broadcast map

# In[40]:


# artistAlias.take(1)


# In[41]:


# print(bArtistAlias.value.get(3990103102312))


# In[42]:


# print(bArtistAlias.value.get(1092764))


# In[11]:


def replaceAlias(line):
    """Replaces alias with canonical artist id"""
    userID, artistID, play_count = [int(ele) for ele in line.split()]
    canonicalArtistID = bArtistAlias.value.get(artistID)
    
    # If this artist id is mapped to a canonical artist id, use that, else this is already a canonical artist id
    if(canonicalArtistID is not None):
        artistID = canonicalArtistID
    
    return Rating(userID, artistID, play_count)


# In[12]:


trainData = rawUserArtistData.map(lambda line: replaceAlias(line)).cache()


# In[45]:


# trainData.take(1)


# In[13]:


model = ALS.trainImplicit(
    ratings = trainData, 
    rank = 10, 
    iterations = 5, 
    lambda_ = 0.01, 
    alpha = 1.0)


# In[24]:


model


# #### Save the model

# In[66]:


model.save(sc, 'collaborative_model')


# In[30]:


model.userFeatures().first()


# #### Verify that #latent features = 10

# In[34]:


len(model.userFeatures().first()[1])


# In[89]:


rawArtistsForUser = rawUserArtistData.map(lambda line: line.split(' '))                                      .filter(lambda tokens: int(tokens[0]) == 2093760)


# ### Get unique artist IDs from the above RDD

# In[90]:


uniqueArtists = set(rawArtistsForUser.map(lambda tokens: int(tokens[1])).collect())


# In[91]:


uniqueArtists


# In[92]:


uniqueArtistsNames = id2name.filter(lambda (artist_id, artist_name): artist_id in uniqueArtists)                             .map(lambda (artist_id, artist_name): artist_name)


# #### Same artists as in book

# In[93]:


uniqueArtistsNames.collect()


# In[94]:


recommendations = model.call("recommendProducts", 2093760, 5)


# In[95]:


uniqueRecAristIDs = set(map(lambda x: x.product, recommendations))


# In[96]:


uniqueRecAristIDs


# #### Get artist names from recommended Artist IDs

# In[97]:


uniqueRecArtistNames = id2name.filter(lambda (aid, aname): aid in uniqueRecAristIDs)                               .map(lambda (aid, aname): aname)


# In[98]:


uniqueRecArtistNames.collect()

