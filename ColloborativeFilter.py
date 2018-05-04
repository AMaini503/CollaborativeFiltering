rawUserArtistData = sc.textFile('audio_data/user_artist_data.txt')
rawUserArtistData.getNumPartitions()
userIDs = rawUserArtistData.map(lambda line: float(line.split()[0]))
artistIDs = rawUserArtistData.map(lambda line: float(line.split()[1]))
rawArtistData = sc.textFile('audio_data/artist_data.txt')

def getArtistIDAndName(line):
    """Gets artist id and name from a line in artist_data.txt"""
    
    # split about tab
    tokens = line.split('\t')
    
    try:
        return [int(tokens[0]), tokens[1].strip()]
    except Exception as e:
        return None

id2name = rawArtistData.map(lambda line: getArtistIDAndName(line))
id2name = id2name.filter(lambda ele: ele is not None)
rawArtistAlias = sc.textFile('audio_data/artist_alias.txt')
artistAlias = rawArtistAlias.map(lambda line: MapAliasToCanonical(line))
artistAlias = artistAlias.filter(lambda ele: ele is not None)

from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

bArtistAlias = sc.broadcast(artistAlias.collectAsMap())

def replaceAlias(line):
    """Replaces alias with canonical artist id"""
    userID, artistID, play_count = [int(ele) for ele in line.split()]
    canonicalArtistID = bArtistAlias.value.get(artistID)
    
    # If this artist id is mapped to a canonical artist id, use that, else this is already a canonical artist id
    if(canonicalArtistID is not None):
        artistID = canonicalArtistID
    
    return Rating(userID, artistID, play_count)


trainData = rawUserArtistData.map(lambda line: replaceAlias(line)).cache()

model = ALS.trainImplicit(
    ratings = trainData, 
    rank = 10, 
    iterations = 5, 
    lambda_ = 0.01, 
    alpha = 1.0)

# get recommendations for the userid given in book
recommendations = model.call("recommendProducts", 2093760, 5)

# make this list ordered to allow zipping
uniqueRecAristIDs = list(set(map(lambda x: x.product, recommendations)))
uniqueRecArtistNames = id2name.filter(lambda (aid, aname): aid in uniqueRecAristIDs).map(lambda (aid, aname): aname)


# write artistID, artistname to a file
with open('results.txt', 'w+') as f:
    results = zip(uniqueRecAristIDs, uniqueRecArtistNames.collect())
    for result in results:
        f.write('%s\n' % str(result))
