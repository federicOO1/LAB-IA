# Segmentazione Semantica - Potsdam Dataset

Nella repo GitHub ci sono tutti i  file .py che ho usato per il progetto. Ho salvato anche tutti i modelli nelle varie fasi del completamento del progetto nella cartella _models_ (ancora devo aggiungerla su github). 

## Baseline

In _baseline_ ci sono tutti i tentativi fatti con diversi learning rate senza usare uno scheduler che lo modificasse con l'avanzare delle epoche.
Per ogni modello sono salvate anche la val_loss e train_loss e un file cdv con tutte le metriche, seguendo questo oridine:
- modello 1: lr=1e-1
- modello 2: lr=1e-2
-
- modello 5: lr=1e-5

I modelli 1 e 2 li ho subito scartati visto che la loss di validazione era altissima.
Ho trainato per 5 epoche i modelli 3,4,5. I risultati sono salvati come 3_2, 4_2, 5_2.

Analizzando le metriche ho deciso di usare come lr iniziale 1e-3.


## Final model

Ho trainato il modello per 50 epoche, aggiungendo come optimzer Adam e uno scheduler del lr. Ho visto che che la train_loss e la val_loss continuavano a scendere, fino quasi a stabilizzarsi verso la fine.

L'IoU aveva alcune classi che presentavano dei valori molto minori delle altre, penso sia dovuto alle immagini usate per effettuare il train. Tutte le metriche di validazione sono salvate nel rispettivo file .csv

## Test

Ho usato un gruppo di immagini per ottenere le predizioni e testare il modello. Come si vedeva dalla validazione ci sono delle classi che vengono predette meno di altre. L'accuracy delle predizioni sul set di test è in media circa tra il 75% e 80%
