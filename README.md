Progetto di tesi magistrale.

Per un’azienda, la capacità di prevedere accuratamente le vendite è fondamentale per ottimizzare
la gestione delle risorse e ridurre al minimo gli sprechi. Prevedere la domanda futura permette di
calibrare la produzione e l’inventario in modo da evitare sia l’accumulo eccessivo di scorte, che può
portare a costi inutili, sia la carenza di prodotti, che può significare vendite perse e insoddisfazione
dei clienti.
L’azienda ORS GROUP, presso la quale ho svolto il progetto di tesi, offre soluzioni cross-industry
per ottimizzare e automatizzare i processi aziendali utilizzando algoritmi proprietari di intelligenza
artificiale, apprendimento automatico e analisi dei Big Data ed è stata fondata in Italia nel 1997.
ORS era interessata a capire se l’aggiunta di immagini e descrizioni dei prodotti ai consueti dati
tabulari utilizzati per la predizione vendite potesse aumentarne l’accuratezza.
L’oggetto di studio per verificare questa ipotesi è stata l’azienda Mauli, per la quale ORS aveva
già realizzato un modello di predizione con soli dati tabulari. Mauli nasce nel 1964 come azienda
produttrice di abbigliamento per bambini e si è rivolta ad ORS perchè ha riconosciuto di avere
difficoltà nella definizione puntuale e corretta delle quantità da acquistare e nell’attribuzione
dell’acquistato ai negozi. L’obbiettivo, dunque, era quello di migliorare il processo di acquisto
e assegnazione della merce prodotta ai punti vendita, arrivando a diminuire le vendite perse e,
conseguentemente, aumentare i margini di fatturato.
Il progetto di tesi prevede la progettazione e l’addestramento di una rete neurale per la predizione
delle vendite dei prodotti di abbigliamento. Essa prenderà in input dati di diverso tipo: le informazioni
contenute nelle immagini verranno estratte tramite un Vision Transformer, quelle contenute
nelle descrizioni, invece, verranno ottenute tramite DistilBERT, versione più leggera della rete
Transformer BERT di Google, presente su Hugging Face. Gli embedding dei due verranno concatenati
alle transazioni di vendita contenute nei dati tabulari. L’input così ottenuto viene poi
fornito ad una rete di tipo feed-forward per prevedere la quantit`a venduta del prodotto. Infine, si
verificheranno le metriche di performance, arrivando a concludere che l’utilizzo delle immagini e
delle descrizioni accresce sensibilmente l’accuratezza della predizione.
