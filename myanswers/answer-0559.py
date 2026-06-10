from sklearn.ensemble import IsolationForest

def detectar_anomalias(input, output=None):
    modelo = IsolationForest()
    modelo.fit(input)
    modelo.predict(input)
