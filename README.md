# OBD

Il progetto è stato realizzato in jupyter lab e poi convertito in python. Le librerie necessarie sono sklearn, numpy, pandas, matplotlib.

Se si ha installato jupyter lab è sufficiente premere la doppia freccia per eseguire interamente il codice. Siccome per la cross validation del problema di classificazione il tempo di esecuzione è circa un giorno la riga è stata commentata. Senza, il tempo di esecuzione è di 14 min. Se si vuole effettuare anche la ricerca dei parametri per la classificazione, dalla cella 18 togliete l’asterisco prima di best_params.

Per mostrare direttamente i risultati dell’esecuzione si può usare Jupyter lab online. Bisogna cercare jupyter lab e cliccare su try poi jupyter lab e si verra reindirizzati alla versione lite. Trascinare il file “.ipynb” sulla colonna di sinistra e cliccarci sopra. Per qualche difetto della versione lite è necessario cliccate su una cella e cambiare da code a markdown (cliccare code ˅ situato sopra il codice e poi selezionare markdown) e poi tornare a code. A quel punto saranno mostrati tutti i risultati e grafici. Se si vuole rieseguire il codice, bisogna trascinare anche i dataset e poi cliccare 2 volte sulla doppia freccia, (controllare che in alto a desta sia presente il kernel), non è consigliato l’esecuzione in questa modalità poiché ci mette 2 ore circa.

Se invece si vuole eseguire il codice python c’è un interfaccia che chiede cosa eseguire.

Qualora non si usasse un IDE, se necessario per installare le librerie i comandi sono

Su linux:

```
python -m venv .venv
source .venv/bin/activate
pip install scikit-learn matplotlib pandas numpy
```

Su Windows

```
python -m venv .venv
./venv/bin/activate.ps1
pip install scikit-learn matplotlib pandas numpy
```

In entrambi i casi è poi necessario eseguire: 

```
python Main.py
```
