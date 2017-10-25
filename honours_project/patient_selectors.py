"""
    Select subsets of patients for cross validation
    based on characteristics.
"""

def selectNoSpeechDisorderPatients(patients):
    test_patients = []
    if type(patients) is dict:
        patients = list(patients.values())

    for pat in patients:
        if pat["hasSpeechDisorder"] == False:
            test_patients.append(pat)
    return test_patients


def selectAllPatients(patients):
    test_patients = []
    if type(patients) is dict:
        patients = list(patients.values())
    for pat in patients:
        test_patients.append(pat)
    return test_patients
