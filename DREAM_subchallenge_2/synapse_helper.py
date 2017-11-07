import synapseclient
import syncredentials
import pickle

def getAndDownloadSynapse():
    syn = synapseclient.Synapse()
    syn.login(syncredentials.username, syncredentials.password)

    tap = syn.tableQuery("select * from syn5511439")
    walk = syn.tableQuery("SELECT * FROM syn5511449")
    talk = syn.tableQuery("SELECT * FROM syn5511444")
    mem = syn.tableQuery("SELECT * FROM syn5511434")
    updrs = syn.tableQuery("SELECT * FROM syn5511432")
    demog = syn.tableQuery("SELECT * FROM syn5511429")
    pdq8 = syn.tableQuery("SELECT * FROM syn5511433")
    print("start")
    taps =  syn.downloadTableColumns(tap, ['tapping_results.json.TappingSamples', 'accel_tapping.json.items'])
    print("tapped")
    walks =  syn.downloadTableColumns(walk, ["accel_walking_outbound.json.items","deviceMotion_walking_outbound.json.items","pedometer_walking_outbound.json.items","accel_walking_return.json.items","deviceMotion_walking_return.json.items","pedometer_walking_return.json.items","accel_walking_rest.json.items","deviceMotion_walking_rest.json.items"])
    print("walked")
    talks =  syn.downloadTableColumns(talk, ["audio_audio.m4a","audio_countdown.m4a"])
    print("talked")
    mems =  syn.downloadTableColumns(mem, ["MemoryGameResults.json.MemoryGameGameRecords"])\

    talk = talk.asDataFrame()
    walk = walk.asDataFrame()
    tap = tap.asDataFrame()
    mem = mem.asDataFrame()
    pdq8 = pdq8.asDataFrame()
    demog = demog.asDataFrame()
    updrs = updrs.asDataFrame()
    with open('syns.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([tap, taps, walk, walks, talk, talks, mem, mems, pdq8, demog, updrs], f)


def getDreamChallengeData():
    syn = synapseclient.Synapse()
    syn.login(syncredentials.username, syncredentials.password)
    walk = syn.tableQuery("SELECT * FROM syn10146553")
    walks =  syn.downloadTableColumns(walk, ["accel_walking_outbound.json.items","deviceMotion_walking_outbound.json.items","pedometer_walking_outbound.json.items","accel_walking_return.json.items","deviceMotion_walking_return.json.items","pedometer_walking_return.json.items","accel_walking_rest.json.items","deviceMotion_walking_rest.json.items"])
    demog = syn.tableQuery("SELECT * FROM syn10146552")

    walk = walk.asDataFrame()
    demog = demog.asDataFrame()
    with open('dream_syns.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([walk, walks, demog], f)


def getDreamChallengeTest():
    syn = synapseclient.Synapse()
    syn.login(syncredentials.username, syncredentials.password)
    walk = syn.tableQuery("SELECT * FROM syn10733842")
    walks =  syn.downloadTableColumns(walk, ["accel_walking_outbound.json.items","deviceMotion_walking_outbound.json.items","pedometer_walking_outbound.json.items","accel_walking_return.json.items","deviceMotion_walking_return.json.items","pedometer_walking_return.json.items","accel_walking_rest.json.items","deviceMotion_walking_rest.json.items"])

    walk = walk.asDataFrame()
    with open('dream_syns_test.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([walk, walks], f)


def getDreamChallengeSupplementary():
    syn = synapseclient.Synapse()
    syn.login(syncredentials.username, syncredentials.password)
    walk = syn.tableQuery("SELECT * FROM syn10733835")
    walks =  syn.downloadTableColumns(walk, ["accel_walking_outbound.json.items","deviceMotion_walking_outbound.json.items","pedometer_walking_outbound.json.items","accel_walking_return.json.items","deviceMotion_walking_return.json.items","pedometer_walking_return.json.items","accel_walking_rest.json.items","deviceMotion_walking_rest.json.items"])

    walk = walk.asDataFrame()
    with open('dream_syns_supp.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([walk, walks], f)

# getDreamChallengeSupplementary()

def restoreSynapseTables():
    with open('syns.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
        return pickle.load(f)


def restoreSynapseTablesDream():
    with open('dream_syns.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
        return pickle.load(f)

def restoreSynapseTablesDreamTest():
    with open('dream_syns_test.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
        return pickle.load(f)

def restoreSynapseTablesDreamSupplementary():
    with open('dream_syns_supp.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
        return pickle.load(f)


def getLDopaChallengeTrain():
    syn = synapseclient.Synapse()
    syn.login(syncredentials.username, syncredentials.password)
    ldopa = syn.tableQuery("SELECT * FROM syn10495809")
    ldopa_files =  syn.downloadTableColumns(ldopa, ["dataFileHandleId"])
    ldopa = ldopa.asDataFrame()
    with open('dream_ldopa_train.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([ldopa, ldopa_files], f)

def getLDopaChallengeTest():
    syn = synapseclient.Synapse()
    syn.login(syncredentials.username, syncredentials.password)
    ldopa = syn.tableQuery("SELECT * FROM syn10701954")
    ldopa_files =  syn.downloadTableColumns(ldopa, ["dataFileHandleId"])
    ldopa = ldopa.asDataFrame()
    with open('dream_ldopa_test.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([ldopa, ldopa_files], f)


def restoreLDopaChallengeTrain():
    with open('dream_ldopa_train.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
        return pickle.load(f)


def restoreLDopaChallengeTest():
    with open('dream_ldopa_test.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
        return pickle.load(f)
