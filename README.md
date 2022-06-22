# arimaModel
Arima model with stoks S&amp;P500. Values: for each day: x = the sum of the 5 securities with the highest increase + the sum of the 5 securities with the worst decrease

Modello autoregressivo integrato a media mobile (ARIMA)
https://it.wikipedia.org/wiki/Modello_autoregressivo_integrato_a_media_mobile

In statistica per modello ARIMA (acronimo di AutoRegressive Integrated Moving Average) si intende una particolare tipologia di modelli atti ad indagare serie storiche che presentano caratteristiche particolari. Fa parte della famiglia dei processi lineari non stazionari.

Un modello ARIMA(p,d,q) deriva da un modello ARMA(p,q) a cui sono state applicate le differenze di ordine d per renderlo stazionario. In caso di stagionalità nei dati si parla di modelli SARIMA o ARIMA(p,d,q)(P,D,Q).


Modello autoregressivo a media mobile (ARMA)
https://it.wikipedia.org/wiki/Modello_autoregressivo_a_media_mobile

Il modello autoregressivo a media mobile, detto anche ARMA, è un tipo di modello matematico lineare che fornisce istante per istante un valore di uscita basandosi sui precedenti valori in entrata e in uscita. A volte denominato modello di Box-Jenkins dal nome dei suoi inventori George Box e Gwilym Jenkins, viene utilizzato in statistica per lo studio delle serie storiche dei dati e in ingegneria dei sistemi nella modellizzazione soprattutto di sistemi meccanici, idraulici o elettronici.

Caratteristiche
Si considera il sistema da descrivere come un'entità che, istante per istante, riceve un valore in entrata (input) e ne genera uno in uscita (output), calcolati in base a dei parametri interni che variano a loro volta in base a leggi lineari. Ogni parametro interno, dunque, verrà ad ogni istante posto uguale a una combinazione lineare di tutti parametri interni dell'istante precedente e del valore in entrata, e il valore in uscita sarà a sua volta una combinazione lineare dei parametri interni e in rari casi anche di quello in entrata; in tal caso si parla di modello improprio, la cui caratteristica principale è di rispondere istantaneamente alle variazioni dell'input e dare luogo a anomalie nel calcolo qualora fosse collegato ad anello con altri sistemi impropri.

Algebricamente, i valori in ingresso e in uscita in un dato istante sono due scalari e i parametri interni formano un vettore. Lo scalare in uscita è il prodotto tra il vettore dei parametri e un vettore fisso c facente parte del modello e di dimensione uguale al numero dei parametri n, sommato all'ingresso moltiplicata per un coefficiente d che nei sistemi impropri è diverso da 0. Il vettore dei parametri è in ogni istante calcolato come la somma dello scalare in ingresso per un vettore b e il precedente vettore dei parametri moltiplicato per una matrice A.

Linearità
Un modello ARMA ha diverse caratteristiche che lo rendono semplice da analizzare:

linearità: moltiplicando tutti i valori in ingresso per un fattore k anche l'uscita risulterà moltiplicata per tale valore. Sommando due sequenze di valori in input si otterrà in output la somma delle sequenze di output che si sarebbero ottenute fornendo i due input indipendentemente.

tempo invarianza: una certa sequenza in input darà una certa sequenza in output indipendentemente dalla quantità di istanti trascorsi dall'istante zero. Lo stesso concetto di "istante zero" è puramente convenzionale poiché il sistema tende a "dimenticare" il passato, ossia ad esserne influenzato in maniera esponenzialmente decrescente nel corso del tempo (caratteristica detta "evanescenza").
Data una serie storica di valori di Xt , il modello di ARMA è uno strumento per analizzare e predire dei valori futuri e consiste di due parti, ossia una parte autoregressiva (AR) e di una parte di media mobile (MA). Il modello è solitamente indicato con ARMA (p,q) dove p è l'ordine della parte autoregressiva e q è l'ordine della parte media mobile.
